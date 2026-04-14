.PHONY: setup download-weights download-sim server run repl eval help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Run initial setup (conda envs, .env template)
	bash tools/setup.sh

download-weights: ## Download OpenVLA-UAV weights from HuggingFace
	python tools/download_weights.py

download-sim: ## Download Unreal simulator and textures from ModelScope
	python tools/download_simulator.py

server: ## Start OpenVLA inference server (requires rvln-server env + GPU)
	python scripts/start_server.py

run: ## Run integrated LTL + diary pipeline (requires rvln-sim env)
	python scripts/run_integration.py

repl: ## Interactive REPL for drone commands
	python scripts/run_repl.py

eval: ## Run UAV-Flow batch evaluation
	python scripts/run_eval.py

lint: ## Check for import errors
	python -c "import rvln; import gym_unrealcv; print('All imports OK')"
