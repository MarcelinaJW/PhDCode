close("*");
dir1 = "/Users/mw4217/Desktop/Marcelina/preprocessed/";
dir2 = "/Users/mw4217/Desktop/Marcelina/postprocessed/";

list = getFileList(dir1);
setBatchMode(true);

for (i = 0; i < lengthOf(list); i++) {
	current_imagePath = dir1+list[i];
	if (File.isDirectory(current_imagePath));{
		open(current_imagePath);
		setBatchMode("show");
		//run("Channels Tool...");	
		
		setBatchMode("show");
			run("Blue");
			//run("Brightness/Contrast...");
			waitForUser ("Brightness and contrast OK?");
			saveAs("DAPI.jpg");
			
			setBatchMode("show");
			run("Cyan");
			//run("Brightness/Contrast...");
			waitForUser ("Brightness and contrast OK?");
			saveAs("BD.jpg");


			setBatchMode("show");
			run("Red");
			//run("Brightness/Contrast...");
			waitForUser ("Brightness and contrast OK?");
			saveAs("Cy3.jpg");
			
			
			setBatchMode("show");
			run("Green");
			//run("Brightness/Contrast...");
			waitForUser ("Brightness and contrast OK?");
			saveAs("BD.jpg");
				
				close();
			}
        }  
		
	

