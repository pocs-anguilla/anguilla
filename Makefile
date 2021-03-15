# Please note that this Makefile assumes a Linux-based development environment.
# For example, that from the provided devcontainer for VS Code.

# For development, prefer the C++20 toolchain, as it has template support for the nodiscard attribute.
TOOLCHAIN_FILE=cmake/toolchains/clang-cxx20-libstdcxx.cmake

COMMON_CMAKE_OPTS=-DWITH_SHARK_BINDINGS=ON -DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE)

cxx_extension: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release ..
	CCACHE_DIR=$(PWD)/_ccache CCACHE_COMPILERCHECK=content make -C _cxx_build -j $(shell nproc)
	cp _cxx_build/anguilla/cxx/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/_dominance.cpython-38-x86_64-linux-gnu.so anguilla/dominance/
	cp _cxx_build/anguilla/cxx/_archive.cpython-38-x86_64-linux-gnu.so anguilla/archive/
	cp _cxx_build/anguilla/cxx/_optimizers.cpython-38-x86_64-linux-gnu.so anguilla/optimizers/
	cp _cxx_build/compile_commands.json compile_commands.json

cxx_extension_debug: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug ..
	make -C _cxx_build -j $(shell nproc)
	cp _cxx_build/anguilla/cxx/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/_dominance.cpython-38-x86_64-linux-gnu.so anguilla/dominance/
	cp _cxx_build/anguilla/cxx/_archive.cpython-38-x86_64-linux-gnu.so anguilla/archive/
	cp _cxx_build/anguilla/cxx/_optimizers.cpython-38-x86_64-linux-gnu.so anguilla/optimizers/
	cp _cxx_build/compile_commands.json compile_commands.json

test:
	#python -m unittest
	pytest tests --color=yes --cov=anguilla


test_debug:
	#LD_PRELOAD="libasan.so libubsan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 python -m unittest
	LD_PRELOAD="libasan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0  python -m unittest

jupyter:
	jupyter-lab --allow-root

jupyter_debug:
	#LD_PRELOAD="libasan.so libubsan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 jupyter-lab --allow-root
	LD_PRELOAD="libasan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 jupyter-lab --allow-root

python_debug:
	#LD_PRELOAD="libasan.so libubsan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 python $(SCRIPT)
	LD_PRELOAD="libasan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 python $(SCRIPT)

update_submodules:
	git submodule update --init --recursive

clean:
	git clean -f -d -X

update_references:
	pybtex-format --style=unsrt docs/references.bib REFERENCES.txt
	mv REFERENCES.txt REFERENCES
