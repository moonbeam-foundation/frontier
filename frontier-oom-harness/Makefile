.PHONY: deploy smoke baseline patched report-latest

deploy:
	cd contracts && chmod +x deploy.sh && ./deploy.sh

smoke: deploy
	DURATION_MS=$$((60*60*1000)) pnpm run run:master

baseline: deploy
	pnpm run run:master

patched: deploy
	pnpm run run:patched

report-latest:
	@LATEST=$$(ls -1 runs 2>/dev/null | grep -v '^\.' | grep -v '^\.gitignore$$' | tail -1); \
	if [ -z "$$LATEST" ]; then echo "No runs found"; exit 1; fi; \
	pnpm run report -- $$LATEST
