#!/bin/bash


echo "Tabula Muris ---------------------------------------------------"
echo "ProtoNet -------------------------------------------------------"


echo "ProtoNet 5-way 1-shot Tabula Muris"
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNet 10-way 1-shot Tabula Muris"
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNet 5-way 5-shot Tabula Muris"
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNet 10-way 5-shot Tabula Muris"
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64


echo "ProtoNetSOT ------------------------------------------------------"


echo "ProtoNetSOT 5-way 1-shot Tabula Muris"
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNetSOT 10-way 1-shot Tabula Muris"
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNetSOT 5-way 5-shot Tabula Muris"
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64

echo "ProtoNetSOT 10-way 5-shot Tabula Muris"
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[16,16] model=FCNet2x16
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[32,32] model=FCNet2x32
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=tm_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=tabula_muris backbone.layer_dim=[64,64,64] model=FCNet3x64


echo "SwissProt ------------------------------------------------------"
echo "ProtoNet -------------------------------------------------------"


echo "ProtoNet 5-way 1-shot SwissProt"
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_backbones_1s5w  n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNet 10-way 1-shot SwissProt"
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_backbones_1s10w  n_way=10 n_shot=1  method=protonet dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNet 5-way 5-shot SwissProt"
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_backbones_5s5w  n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNet 10-way 5-shot SwissProt"
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_backbones_5s10w  n_way=10 n_shot=5  method=protonet dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128


echo "ProtoNetSOT ------------------------------------------------------"


echo "ProtoNetSOT 5-way 1-shot SwissProt"
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_sot_backbones_1s5w  n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNetSOT 10-way 1-shot SwissProt"
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_sot_backbones_1s10w  n_way=10 n_shot=1  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNetSOT 5-way 5-shot SwissProt"
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_sot_backbones_5s5w  n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128

echo "ProtoNetSOT 10-way 5-shot SwissProt"
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64] model=FCNet2x64
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128] model=FCNet2x128
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[256,256] model=FCNet2x256
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[512,512] model=FCNet2x512
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[64,64,64] model=FCNet3x64
python run.py exp.name=sp_protonet_sot_backbones_5s10w  n_way=10 n_shot=5  method=protonet_sot dataset=swissprot backbone.layer_dim=[128,128,128] model=FCNet3x128
