# Please note that this Makefile assumes a Linux-based development environment.
# For example, that from the provided devcontainer for VS Code.

# For development, prefer the C++20 toolchain, as it has template support for the nodiscard attribute.
TOOLCHAIN_FILE=cmake/toolchains/clang-cxx20-libstdcxx.cmake
#TOOLCHAIN_FILE=cmake/toolchains/clang-cxx20-libcxx.cmake

COMMON_CMAKE_OPTS=-DWITH_SHARK_BINDINGS=ON -DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN_FILE)

cxx_extension: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_BUILD_TYPE=Release ..
	make -C _cxx_build -j $(nproc)
	cp _cxx_build/anguilla/cxx/hypervolume/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/shark_hypervolume/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/dominance/_dominance.cpython-38-x86_64-linux-gnu.so anguilla/dominance/
	cp _cxx_build/anguilla/cxx/archive/_archive.cpython-38-x86_64-linux-gnu.so anguilla/archive/

cxx_extension_debug: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_BUILD_TYPE=Debug ..
	make -C _cxx_build -j $(nproc)
	cp _cxx_build/anguilla/cxx/hypervolume/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/shark_hypervolume/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/dominance/_dominance.cpython-38-x86_64-linux-gnu.so anguilla/dominance/
	cp _cxx_build/anguilla/cxx/archive/_archive.cpython-38-x86_64-linux-gnu.so anguilla/archive/

cxx_experiments: notebooks/shark/exploration/CMakeLists.txt
	mkdir -p _cxx_experiments_build
	cd _cxx_experiments_build ;\
	set -x; cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/clang-cxx14-libstdcxx.cmake -DCMAKE_BUILD_TYPE=Release ../notebooks/shark/exploration
	make -C _cxx_experiments_build -j $(nproc)

test:
	python -m unittest

test_debug:
	#LD_PRELOAD="libasan.so libubsan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 python -m unittest
	LD_PRELOAD="libasan.so" ASAN_OPTIONS=check_initialization_order=1 ASAN_OPTIONS=detect_leaks=0 python -m unittest

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
