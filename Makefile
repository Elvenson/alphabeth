# Makefile for TStream

# Definition
SHELL := /bin/bash # use bash all the time!
FILES = $(shell find . -type f -name "*.go" | grep -v "vendor")

# `make help` for more
help: ## This is help dialog.
help h:
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%-20s %s\n" "target" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done


lint:	## Lint the codebase.
	@golint -set_exit_status ./...

gotest: ## Go test codebase.
	go test ./...

fmt:	## Go fmt package
	for FILE in $(FILES); do \
  		go fmt $$FILE; \
    done

ci: lint gotest	## Simulate gitlab-CI
	@echo "CI passed!"
