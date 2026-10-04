close("*");
dir1 = "/Users/mw4217/Desktop/Marcelina/preprocessed/";
dir2 = "/Users/mw4217/Desktop/Marcelina/postprocessed/";

list = getFileList(dir1);
setBatchMode(true);

for (i = 0; i < lengthOf(list); i++) {
	current_imagePath = dir1+list[i];
	open(current_imagePath);
	setBatchMode("show");
	run("Colour Deconvolution", "vectors=[H DAB]");
    close();
   	//run("Threshold...");
	selectImage(list[i]+"-(Colour_2)");
	setAutoThreshold("Default dark no-reset");
	//set the threshold & change for each region
	setThreshold(0, 50, "raw");
	setOption("BlackBackground", true);
	run("Convert to Mask");
	run("Measure");
	setResult("Name", i, substring(list[i], 0, indexOf(list[i], ".")));
	close("*");
	
}  

saveAs("Measurements", dir2 +"Results.csv");
selectWindow("Results");
run("Close");
	


			