.PHONY: build test lint clippy deny clean coverage

build:
	cargo build

test:
	cargo test

lint: clippy deny

clippy:
	cargo clippy -- -D warnings

deny:
	cargo deny check

coverage:
	cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

clean:
	cargo clean
