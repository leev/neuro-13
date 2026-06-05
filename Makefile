.PHONY: html watch run-all check pages deploy

html:
	python3 build.py

# Refresh the Cloudflare Pages deploy directory from the source files.
pages: html
	mkdir -p public
	cp neuro-13.html public/index.html
	cp neuro1.py neuro2.py neuro3.py neuro4.py neuro5.py \
	   neuro6.py neuro7.py neuro8.py neuro9.py public/

# Deploy public/ to Cloudflare as a Worker with the Assets binding (see wrangler.toml).
# Worker is bound to the custom domain only — no *.workers.dev or *.pages.dev URL.
deploy: pages
	npx wrangler@latest deploy

# Re-run build whenever a .py file changes (requires `fswatch` on macOS).
watch:
	@command -v fswatch >/dev/null || { echo "install fswatch: brew install fswatch"; exit 1; }
	@fswatch -o neuro*.py | xargs -n1 -I{} python3 build.py

# Run all the non-PyTorch scripts (1-7) end to end.
run-all:
	@for f in neuro1.py neuro2.py neuro3.py neuro4.py neuro5.py neuro6.py neuro7.py; do \
		echo "--- $$f ---"; \
		python3 $$f; \
		echo; \
	done

# Sanity check: every script at least parses.
check:
	@for f in neuro*.py; do python3 -c "import ast; ast.parse(open('$$f').read())" && echo "$$f OK"; done
