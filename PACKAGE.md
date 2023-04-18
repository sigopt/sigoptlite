<!--
Copyright © 2023 Intel Corporation

SPDX-License-Identifier: Apache License 2.0
-->

## Build

1. Update version in `setup.cfg`
2. Create new branch and push to Github
3. Ensure nothing is left from old builds: `rm -rf dist; rm -rf sigoptlite.egg-info/`
4. Build package: `python -m build`

## Upload

1. `twine upload dist/*`
2. Follow twine prompts to upload
3. Tag released code on github and merge into main
     ex: `git tag -a v0.0.1 -m "code for v0.0.1"` and push: `git push origin tag v0.0.1`
