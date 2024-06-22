from os import listdir, path


def get_all_files(target_dir: str):
	assert path.exists(target_dir), f"Directory {target_dir} does not exist"
	filepaths = listdir(target_dir)
	return list(map(lambda fp: target_dir + "/" + fp, filepaths))
