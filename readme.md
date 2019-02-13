# pyhej-gans
因为包含子模块：
```
git submodule init
git submodule update
```

## fastMRI
通过获得更少的测量来加速磁共振成像（MRI）有可能降低医疗成本，最大限度地减少对患者的压力，并使MR成像在目前非常缓慢或昂贵的应用中成为可能。[github](https://github.com/facebookresearch/fastMRI)
```
git submodule add https://github.com/facebookresearch/fastMRI.git module_fast_mri
git submodule update --remote
```

移除子模块：
```
git rm --cached module_fast_mri
rm -rf module_fast_mri
```
