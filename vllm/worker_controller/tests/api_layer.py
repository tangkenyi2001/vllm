# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

LogEntry = tuple[float, str]


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run a single prompt through vLLM OpenAI API server.",
	)
	parser.add_argument(
		"--model",
		default="facebook/opt-125m",
		help="Model name, e.g. facebook/opt-125m",
	)
	parser.add_argument("--prompt", default="Hello, my name is", help="Input prompt")
	parser.add_argument("--host", default="127.0.0.1", help="API server host")
	parser.add_argument("--port", type=int, default=8010, help="API server port")
	parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
	parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top-p")
	parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
	parser.add_argument(
		"--gpu-memory-utilization",
		type=float,
		default=0.5,
		help="GPU memory utilization to pass to API server",
	)
	parser.add_argument(
		"--enforce-eager",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Enable eager mode in API server (default: enabled)",
	)
	parser.add_argument(
		"--startup-timeout",
		type=float,
		default=180.0,
		help="Seconds to wait for API server health check",
	)
	parser.add_argument(
		"--show-api-logs",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Show API server stdout/stderr while running (default: enabled)",
	)
	return parser


def _collect_logs(process: subprocess.Popen, show_logs: bool) -> tuple[list[LogEntry], threading.Thread]:
	log_lines: list[LogEntry] = []

	def _reader() -> None:
		if process.stdout is None:
			return
		for raw_line in process.stdout:
			line = raw_line.rstrip("\n")
			if not line:
				continue
			log_lines.append((time.time(), line))
			if len(log_lines) > 5000:
				del log_lines[:1000]
			if show_logs:
				print(f"[api-server] {line}", flush=True)

	t = threading.Thread(target=_reader, daemon=True)
	t.start()
	return log_lines, t


def _extract_log_timestamp_seconds(line: str) -> float | None:
	match = re.search(r"\b\d{2}-\d{2} (\d{2}):(\d{2}):(\d{2})\b", line)
	if not match:
		return None
	hour = int(match.group(1))
	minute = int(match.group(2))
	second = int(match.group(3))
	return float(hour * 3600 + minute * 60 + second)


def _parse_startup_timings(log_lines: list[LogEntry]) -> dict[str, float]:
	timings: dict[str, float] = {}

	startup_re = re.compile(
		r"MultiprocExecutor startup breakdown: worker_ready=([0-9.]+)s, mq_ready=([0-9.]+)s, total=([0-9.]+)s"
	)
	weight_re = re.compile(r"Loading weights took\s+([0-9.]+)\s+seconds")
	model_load_re = re.compile(
		r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"
	)
	profile_re = re.compile(r"profiling memory takes:\s*([0-9.]+)s")
	engine_re = re.compile(
		r"init engine \(profile, create kv cache, warmup model\) took\s+([0-9.]+)\s+seconds"
	)

	for _, line in log_lines:
		match = startup_re.search(line)
		if match:
			timings["worker_process_startup_s"] = float(match.group(1))
			timings["mq_setup_s"] = float(match.group(2))
			timings["multiproc_executor_total_s"] = float(match.group(3))
			continue

		match = weight_re.search(line)
		if match:
			timings["model_weight_load_s"] = float(match.group(1))
			continue

		match = model_load_re.search(line)
		if match:
			timings["model_load_total_s"] = float(match.group(1))
			continue

		match = profile_re.search(line)
		if match:
			timings["memory_profile_s"] = float(match.group(1))
			continue

		match = engine_re.search(line)
		if match:
			timings["engine_init_profile_kv_cache_warmup_s"] = float(match.group(1))

	return timings


def _parse_milestones(log_lines: list[LogEntry], spawn_end_ts: float,
					  ready_ts: float) -> dict[str, float]:
	milestones: dict[str, float] = {}

	if log_lines:
		milestones["first_log_after_spawn_s"] = max(0.0, log_lines[0][0] - spawn_end_ts)

	event_patterns = {
		"api_process_started_log": "vLLM API server version",
		"engine_process_init_log": "Initializing a V1 LLM engine",
		"distributed_init_log": "world_size=",
		"model_load_start_log": "Starting to load model",
		"weights_loaded_log": "Loading weights took",
		"model_loaded_log": "Model loading took",
		"kv_cache_init_done_log": "init engine (profile, create kv cache, warmup model) took",
		"api_routes_ready_log": "Starting vLLM API server",
		"uvicorn_started_log": "Started server process",
		"app_startup_complete_log": "Application startup complete",
		"health_ok_log": "GET /health",
	}

	event_clock: dict[str, float] = {}
	event_received: dict[str, float] = {}

	for received_ts, line in log_lines:
		for event_name, pattern in event_patterns.items():
			if event_name in event_clock:
				continue
			if pattern in line:
				clock_ts = _extract_log_timestamp_seconds(line)
				if clock_ts is not None:
					event_clock[event_name] = clock_ts
				event_received[event_name] = received_ts

	for event_name, received_ts in event_received.items():
		milestones[f"{event_name}_after_spawn_s"] = max(0.0, received_ts - spawn_end_ts)

	ordered_event_chain = [
		"api_process_started_log",
		"engine_process_init_log",
		"distributed_init_log",
		"model_load_start_log",
		"weights_loaded_log",
		"model_loaded_log",
		"kv_cache_init_done_log",
		"api_routes_ready_log",
	]

	for prev_name, next_name in zip(ordered_event_chain, ordered_event_chain[1:]):
		if prev_name in event_clock and next_name in event_clock:
			delta = event_clock[next_name] - event_clock[prev_name]
			if delta >= 0:
				milestones[f"{prev_name}__to__{next_name}_s"] = delta

	milestones["ready_after_spawn_s"] = max(0.0, ready_ts - spawn_end_ts)
	return milestones


def _print_timing_breakdown(init_time_s: float, timings: dict[str, float],
						   milestones: dict[str, float] | None = None) -> None:
	def _metric(key: str) -> float | None:
		if milestones is None:
			return None
		value = milestones.get(key)
		return value if isinstance(value, (int, float)) else None

	def _print_metric(name: str, value: float | None) -> None:
		if value is None:
			print(f"  {name}: N/A")
		else:
			print(f"  {name}: {value:.3f}")

	print("Startup timing breakdown:")
	print(f"  total_api_ready_s: {init_time_s:.3f}")

	if not timings:
		print("  parsed_timings: unavailable (no matching vLLM startup lines found)")
		return

	ordered_keys = [
		"worker_process_startup_s",
		"mq_setup_s",
		"multiproc_executor_total_s",
		"model_weight_load_s",
		"model_load_total_s",
		"memory_profile_s",
		"engine_init_profile_kv_cache_warmup_s",
	]
	for key in ordered_keys:
		if key in timings:
			print(f"  {key}: {timings[key]:.3f}")

	engine_total = timings.get("engine_init_profile_kv_cache_warmup_s")
	profile_time = timings.get("memory_profile_s")
	if isinstance(engine_total, (int, float)) and isinstance(profile_time, (int, float)):
		kv_plus_warmup_est = max(0.0, engine_total - profile_time)
		print(f"  kv_cache_plus_warmup_estimated_s: {kv_plus_warmup_est:.3f}")

	known_components = [
		timings.get("model_load_total_s"),
		timings.get("engine_init_profile_kv_cache_warmup_s"),
	]
	known_sum = sum(v for v in known_components if isinstance(v, (int, float)))
	if known_sum > 0:
		remaining = max(0.0, init_time_s - known_sum)
		print(f"  remaining_startup_overhead_s: {remaining:.3f}")

	print("Startup overhead decomposition:")
	api_process_bootstrap_s = _metric("api_process_started_log_after_spawn_s")
	engine_init_seen_s = _metric("engine_process_init_log_after_spawn_s")
	distributed_init_seen_s = _metric("distributed_init_log_after_spawn_s")
	model_load_start_seen_s = _metric("model_load_start_log_after_spawn_s")
	model_load_done_seen_s = _metric("model_loaded_log_after_spawn_s")
	kv_ready_seen_s = _metric("kv_cache_init_done_log_after_spawn_s")
	api_routes_ready_seen_s = _metric("api_routes_ready_log_after_spawn_s")
	health_ready_seen_s = _metric("health_ok_log_after_spawn_s")

	api_bootstrap_until_engine_init_visible_s = None
	if api_process_bootstrap_s is not None and engine_init_seen_s is not None:
		api_bootstrap_until_engine_init_visible_s = max(
			0.0, engine_init_seen_s - api_process_bootstrap_s)

	worker_process_startup_est_s = None
	if engine_init_seen_s is not None and distributed_init_seen_s is not None:
		worker_process_startup_est_s = max(0.0, distributed_init_seen_s - engine_init_seen_s)

	engine_pre_model_overhead_s = None
	if distributed_init_seen_s is not None and model_load_start_seen_s is not None:
		engine_pre_model_overhead_s = max(0.0, model_load_start_seen_s - distributed_init_seen_s)

	engine_creation_phase_s = None
	if engine_init_seen_s is not None and kv_ready_seen_s is not None:
		engine_creation_phase_s = max(0.0, kv_ready_seen_s - engine_init_seen_s)

	model_loading_phase_s = None
	if model_load_done_seen_s is not None and model_load_start_seen_s is not None:
		model_loading_phase_s = max(0.0, model_load_done_seen_s - model_load_start_seen_s)

	post_engine_to_routes_s = None
	if kv_ready_seen_s is not None and api_routes_ready_seen_s is not None:
		post_engine_to_routes_s = max(0.0, api_routes_ready_seen_s - kv_ready_seen_s)

	kv_cache_phase_after_model_load_s = _metric(
		"model_loaded_log__to__kv_cache_init_done_log_s")

	routes_to_health_s = None
	if api_routes_ready_seen_s is not None and health_ready_seen_s is not None:
		routes_to_health_s = max(0.0, health_ready_seen_s - api_routes_ready_seen_s)

	post_engine_api_readiness_s = None
	if kv_ready_seen_s is not None and health_ready_seen_s is not None:
		post_engine_api_readiness_s = max(0.0, health_ready_seen_s - kv_ready_seen_s)

	_print_metric("api_process_bootstrap_to_first_api_log_s", api_process_bootstrap_s)
	_print_metric("api_bootstrap_until_engine_init_visible_s", api_bootstrap_until_engine_init_visible_s)
	_print_metric("  worker_process_startup_estimated_s (included above)", worker_process_startup_est_s)
	_print_metric("  engine_pre_model_overhead_estimated_s (included above)", engine_pre_model_overhead_s)
	_print_metric("engine_creation_phase_s", engine_creation_phase_s)
	_print_metric("  model_loading_phase_s (included above)", model_loading_phase_s)
	_print_metric("  kv_cache_phase_after_model_load_s (included above)", kv_cache_phase_after_model_load_s)
	_print_metric("post_engine_api_readiness_s", post_engine_api_readiness_s)
	_print_metric("  post_engine_to_api_routes_ready_s (included above)", post_engine_to_routes_s)
	_print_metric("  api_routes_to_health_ready_s (included above)", routes_to_health_s)

	if milestones:
		print("Startup milestones (relative timings):")
		priority_order = [
			"first_log_after_spawn_s",
			"api_process_started_log_after_spawn_s",
			"engine_process_init_log_after_spawn_s",
			"distributed_init_log_after_spawn_s",
			"model_load_start_log_after_spawn_s",
			"weights_loaded_log_after_spawn_s",
			"model_loaded_log_after_spawn_s",
			"kv_cache_init_done_log_after_spawn_s",
			"api_routes_ready_log_after_spawn_s",
			"uvicorn_started_log_after_spawn_s",
			"app_startup_complete_log_after_spawn_s",
			"health_ok_log_after_spawn_s",
			"ready_after_spawn_s",
		]
		for key in priority_order:
			if key in milestones:
				print(f"  {key}: {milestones[key]:.3f}")

		segment_keys = [
			"api_process_started_log__to__engine_process_init_log_s",
			"engine_process_init_log__to__distributed_init_log_s",
			"distributed_init_log__to__model_load_start_log_s",
			"model_load_start_log__to__weights_loaded_log_s",
			"weights_loaded_log__to__model_loaded_log_s",
			"model_loaded_log__to__kv_cache_init_done_log_s",
			"kv_cache_init_done_log__to__api_routes_ready_log_s",
		]
		print("Startup milestones (log timestamp segments):")
		for key in segment_keys:
			if key in milestones:
				print(f"  {key}: {milestones[key]:.3f}")


def _wait_for_server(base_url: str, timeout_s: float) -> bool:
	deadline = time.time() + timeout_s
	health_url = f"{base_url}/health"
	while time.time() < deadline:
		try:
			with urlopen(health_url, timeout=2) as response:
				if response.status == 200:
					return True
		except URLError:
			pass
		except Exception:
			pass
		time.sleep(0.5)
	return False


def _post_json(url: str, payload: dict) -> dict:
	body = json.dumps(payload).encode("utf-8")
	request = Request(
		url,
		data=body,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urlopen(request, timeout=120) as response:
		return json.loads(response.read().decode("utf-8"))


def main() -> int:
	args = _build_parser().parse_args()

	server_cmd = [
		sys.executable,
		"-m",
		"vllm.entrypoints.openai.api_server",
		"--model",
		args.model,
		"--host",
		args.host,
		"--port",
		str(args.port),
		"--gpu-memory-utilization",
		str(args.gpu_memory_utilization),
	]
	if args.enforce_eager:
		server_cmd.append("--enforce-eager")

	print(
		f"Starting API server for model {args.model!r} at http://{args.host}:{args.port}...",
		flush=True,
	)
	repo_root = Path(__file__).resolve().parents[3]
	start = time.time()
	spawn_begin = time.time()
	process = subprocess.Popen(
		server_cmd,
		cwd=str(repo_root),
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		text=True,
		bufsize=1,
	)
	spawn_end = time.time()
	print(f"API process spawn call time: {spawn_end - spawn_begin:.3f} seconds", flush=True)
	log_lines, log_thread = _collect_logs(process, args.show_api_logs)

	base_url = f"http://{args.host}:{args.port}"
	print("Waiting for server health check...", flush=True)
	if not _wait_for_server(base_url, args.startup_timeout):
		process.terminate()
		process.wait(timeout=10)
		log_thread.join(timeout=1)
		print(f"Failed to start API server within {args.startup_timeout:.1f} seconds")
		parsed = _parse_startup_timings(log_lines)
		milestones = _parse_milestones(log_lines, spawn_end, start + args.startup_timeout)
		_print_timing_breakdown(args.startup_timeout, parsed, milestones)
		return 1

	init_time_s = time.time() - start
	print(f"API server init time: {init_time_s:.2f} seconds")
	parsed = _parse_startup_timings(log_lines)
	ready_ts = time.time()
	milestones = _parse_milestones(log_lines, spawn_end, ready_ts)
	_print_timing_breakdown(init_time_s, parsed, milestones)

	payload = {
		"model": args.model,
		"prompt": args.prompt,
		"temperature": args.temperature,
		"top_p": args.top_p,
		"max_tokens": args.max_tokens,
	}

	print("Sending completion request...", flush=True)
	generate_start = time.time()
	try:
		response = _post_json(f"{base_url}/v1/completions", payload)
	finally:
		process.terminate()
		process.wait(timeout=10)
		log_thread.join(timeout=1)

	elapsed = time.time() - generate_start
	text = response["choices"][0]["text"] if response.get("choices") else ""
	print(f"Generation time: {elapsed:.2f} seconds")
	print(f"Prompt: {args.prompt!r}, Generated text: {text!r}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
