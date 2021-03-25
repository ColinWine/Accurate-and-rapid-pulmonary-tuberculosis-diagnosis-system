##Accurate and rapid pulmonary tuberculosis diagnosis system
This repo provides the trained AI models and testing code for CT Analysis system described in our paper “A Fully Automatic AI-based CT Image Analysis System for Accurate Detection, Diagnosis, and Quantitative Severity Evaluation of Pulmonary Tuberculosis”.

##Project structure

	|-- src    #Source code
	    |-- Detector
			|-- Detection.py
	    |-- SliceSelection
		    |-- SliceSelection.py		
		|-- LungSeg.py

	|-- Dataset #Folder to put Raw CT scans
	
	|-- Result
	    |-- SelectedSlices   #Folder to put selected slices
		|-- ActivationMap   #Folder to put visualized activation map
	    |-- DetectionResult  #Folder to put visualized detection and classification result
	    |-- Serverity.xls   #Serverity Assessment Result

##Requirements


- python >=3.6
- pytorch 1.1.0 (gpu)
- SimpleITK
- xlwt

##Test

Download pretrained weight at **https://pan.baidu.com/s/18JB_VILkz_upV1X0meNv4A** with code：**m5hh**.

Put the pretrained weight at src\Detector\ckpt\hourglass directory.

1. 
	Put raw CT slices in Dataset folder like this

		|-- Dataset
		    |-- Patient01
	    		|-- slice01.dcm
				|-- slice02.dcm
		    |-- Patient02
	    		|-- slice01.dcm
				|-- slice02.dcm

	 Then run lung segmentation preprocessing program at ./src directory
	```
	cd src
	``` 

	``` 
	python LungSeg.py 
	``` 
	
	Generated lung lobe segmentation mask will be saved at the same folder like this.


		    |-- Dataset
			    |-- Patient01
		    		|-- slice01.dcm
					|-- slice02.dcm
		    		|-- lobes_slice01.nii.gz
					|-- lobes_slice02.nii.gz



2. 
	```
	cd src/SliceSelection   
	```
	and then
	```
	python sliceselection.py
	``` 
	
	Selected slices will be saved at Result/Top10 Slices folder.

	Activation Map will be saved at Result/Activation Map folder.

	Serverity evaluation result will be saved at Result/serverity.xls file

3. 
	```
	cd src/Detection   
	```
	and then
	```
	python Detection.py
	``` 
	
	Visualized detection and classification result will be saved at Result/DetectionResult folder.
	

















