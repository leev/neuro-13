.PHONY: build pages serve smoke check run-all deploy

build:
	python3 build.py

# GitHub Actions calls this target.
pages: build

serve: build
	cd public && python3 -m http.server 8000

smoke: build
	tools/smoke.sh public

# Deploy public/ to Cloudflare as a Worker with the Assets binding (see wrangler.toml).
# The Worker is bound to the custom domain only: no *.workers.dev or *.pages.dev URL.
deploy: build
	npx wrangler@latest deploy

check:
	@for f in py/*.py; do python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" $$f && echo "$$f OK"; done

# Runs every exercise that needs only numpy / scikit-learn (skips the PyTorch ones).
# diffusion.py gets --quick: its full run takes minutes and rewrites the shipped weights.
run-all:
	@cd py && for f in *.py; do \
		if grep -q "^import torch\|^from torch" $$f; then echo "--- $$f (skipped: needs PyTorch) ---"; continue; fi; \
		args=""; if [ $$f = diffusion.py ]; then args="--quick"; fi; \
		echo "--- $$f $$args ---"; python3 $$f $$args || exit 1; echo; \
	done
