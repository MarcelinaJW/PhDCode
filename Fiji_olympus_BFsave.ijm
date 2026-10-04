close("*");
dir1 = getDirectory("/Users/mw4217/Desktop/Marcelina/preprocessed");
dir2 = getDirectory("/Users/mw4217/Desktop/Marcelina/postprocessed");

list = getFileList(dir1);
setBatchMode(true);

for (i = 0; i < lengthOf(list); i++) {
	current_imagePath = dir1+list[i];
	if (File.isDirectory(current_imagePath));{
		open(current_imagePath);
		setBatchMode("show");
		// Create color image.
run("Stack to RGB");
		;
			saveAs("originalfilename_RGB.tiff");
	close();
			}
        }  
		
	


			