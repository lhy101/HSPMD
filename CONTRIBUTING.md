# Contributing to HSPMD
Welcome to [report Issues](https://github.com/PKU-DAIR/HSPMD/issues) or [pull requests](https://github.com/PKU-DAIR/HSPMD/pulls). It's recommended to read the following Contributing Guide first before contributing. 


## Issues
We use Github Issues to track public bugs and feature requests.

### Search Known Issues First
Please search the existing issues to see if any similar issue or feature request has already been filed. You should make sure your issue isn't redundant.

### Reporting New Issues
If you open an issue, the more information the better. Such as detailed description, screenshot or video of your problem, logcat or code blocks for your crash.

## Pull Requests
We strongly welcome your pull request to make HSPMD better. 

### Branch Management
There are three main branches here:

1. `main` branch.

	(1). It is the latest (pre-)release branch. We use `main` for tags, with version number `1.0.0`, `1.1.0`, `1.2.0`...

	(2). **Don't submit any PR on `main` branch.**
	
2. `specific version` branchs. 

	(1).There is a `specific version` for each HSPMD version, such as `branch-1.0.0`, `branch-1.1.0`. It is our stable developing	 branch. After full testing, `specific version` branch will be merged to `main` branch for the next release.

	(2). **You are recommended to submit bugfix or feature PR on `specific version` branch.**


Normal bugfix or feature request should be submitted to `specific version` branch. After full testing, we will merge them to `main` branch for the next release. 


### Make Pull Requests
The code team will monitor all pull request, we run some code check and test on it. After all tests passed, we will accecpt this PR. But it won't merge to `main` branch at once, which have some delay.

Before submitting a pull request, please make sure the followings are done:

1. Fork the repo and create your branch from `main` or `specific version`.
2. Update code or documentation if you have changed APIs.
3. Add the copyright notice to the top of any new files you've added.
4. Check your code lints and checkstyles.
5. Test and test again your code.
6. Now, you can submit your pull request on  `specific version` branch.

## Code Style Guide
Use [Code Style](./.clang-format) for Python and C++.

## License
By contributing to HSPMD, you agree that your contributions will be licensed
under [License](LICENSE)