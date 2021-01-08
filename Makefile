COMMON_CMAKE_OPTS=-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang-cxx20-libcxx.cmake

cxx_extension: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_BUILD_TYPE=Release ..
	make -C _cxx_build -j $(nproc)
	cp _cxx_build/anguilla/cxx/hypervolume/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/shark_hypervolume/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/

cxx_extension_debug: CMakeLists.txt
	mkdir -p _cxx_build
	cd _cxx_build ;\
	set -x; cmake $(COMMON_CMAKE_OPTS) -DCMAKE_BUILD_TYPE=Debug ..
	make -C _cxx_build -j $(nproc)
	cp _cxx_build/anguilla/cxx/hypervolume/_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/
	cp _cxx_build/anguilla/cxx/shark_hypervolume/_shark_hypervolume.cpython-38-x86_64-linux-gnu.so anguilla/hypervolume/

test:
	python -m unittest

test_debug:
	LD_PRELOAD=libasan.so python -m unittest

clean:
	git clean -f -d -X
