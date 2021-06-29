# Note !!: Recommend to test with the model trained in the same condition (GC or pa, preprocessed or not).

# training
python main_multiple.py \
    --DATA_PATH_LIST \
        /data/Users/katafuchi/RA/Nematode/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-18 \
    --MASK_NAME \
        /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_0000.json \
        /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_0000.json \
        /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4_0000.json \
        /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-18_0000.json \
    --SAVE_PATH /data/Users/katafuchi/RA/Nematode/result_multiple \
    --train \
    --kind pa \
    --preprocess \
    --aug_times 100 \
    --epochs 200 

# test
python main_multiple.py \
    --DATA_PATH_LIST \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-2 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-6 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-7 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-8 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-9 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-10 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-12 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-13 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-14 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-15 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-16 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-17 \
        /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-18 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-2 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-6 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-7 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-8 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-9 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-10 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-12 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-13 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-14 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-15 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-16 \
        /data/Users/katafuchi/RA/Nematode/2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-17 \
    --SAVE_PATH /data/Users/katafuchi/RA/Nematode/result_multiple \
    --kind pa \
    --preprocess \
    --model_path /data/Users/katafuchi/RA/Nematode/result_multiple/model.h5\


# """ DATASET NAME LIST

#     '/data/Users/katafuchi/RA/Nematode'

#         '2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3'
    
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-2'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-6'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-7'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-8'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-9'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-10'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-12'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-13'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-14'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-15'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-16'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-17'
#         '2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-18'

#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-2'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-6'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-7'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-8'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-9'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-10'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-12'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-13'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-14'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-15'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-16'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-17'
#         '2021_0324_dia10-5-gcy28d-GCaMP6f+paQuasAr3-18'
    
# """

