# Python interface for QUGaR

Can be installed with 

```bash
python3 -m pip -v install --config-settings=cmake.build-type="Release" --no-build-isolation .
```

If QUGaR cpp library is not found by CMake, its path can be provided as

python3 -m pip -v install --config-settings=cmake.build-type="Release" "--config-settings=cmake.args=-DSOME_DEFINE=ON;-DOTHER=OFF" --no-build-isolation .
```bash
python3 -m pip -v install --config-settings=cmake.build-type="Release" "--config-settings=cmake.args=-DQUGAR_DIR=PATH_TO_QUGAR_INSTALLATION" --no-build-isolation .
```