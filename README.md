# Facial-Recognization-Using-MATLAB
This project aims to develop a system that will automatically capture images (150 images) in a few seconds and then use the image dataset will detect the person in front of the camera.

# Steps
1. Data Collection
2. Train model
3. Test Model

# Code
    
## DATA COLLECTION:
    clc
    clear all
    close all
    warning off;
    cao=webcam;
    faceDetector=vision.CascadeObjectDetector;
    c=150;
    temp=0;
    while true
        e=cao.snapshot;
        bboxes =step(faceDetector,e);
        if(sum(sum(bboxes))~=0)
        if(temp>=c)
            break;
        else
        es=imcrop(e,bboxes(1,:));
        es=imresize(es,[227 227]);
        filename=strcat(num2str(temp),'.bmp');
        imwrite(es,filename);
        temp=temp+1;
        imshow(es);
        drawnow;
        end
        else
            imshow(e);
            drawnow;
        end
    end
## Train model
    clc
    clear all
    close all
    warning off
    g=alexnet;
    layers=g.Layers;
    layers(23)=fullyConnectedLayer(2);
    layers(25)=classificationLayer;
    allImages=imageDatastore('database','IncludeSubfolders',true, 'LabelSource','foldernames');
    opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
    myNet1=trainNetwork(allImages,layers,opts);
    save myNet1;

## Test model
    clc;
    close;
    clear
    c=webcam;
    load myNet1;
    faceDetector=vision.CascadeObjectDetector;
    while true
        e=c.snapshot;
        bboxes =step(faceDetector,e);
        if(sum(sum(bboxes))~=0)
         es=imcrop(e,bboxes(1,:));
        es=imresize(es,[227 227]);
        label=classify(myNet1,es);
        image(e);
        title(char(label));
        drawnow;
        else
            image(e);
            title('No Face Detected');
        end

## Output
<img width="421" height="374" alt="399615990-4ee1c96f-413e-40f1-9338-18e9b04e3ff2" src="https://github.com/user-attachments/assets/7f6b2ed7-6a47-4be1-a12f-75fbf951f6f6" />
<img width="422" height="376" alt="399615988-ab9c92ad-c6ac-4bc2-95d5-48e09d8f5165" src="https://github.com/user-attachments/assets/e7000043-998e-4db4-8b53-9bd21cc098c5" />
<img width="420" height="373" alt="399615984-96df4a5c-7586-4e5c-9158-7edf66dc4927" src="https://github.com/user-attachments/assets/ef124d7d-f8f6-4f42-9f69-c1fd2454206e" />
