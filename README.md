# DIG-Net 
[[project]](https://ppingyes.github.io/digproject/index.html) [[paper]](https://ppingyes.github.io/digproject/paper/jvcir-deep-intensity-guidance.pdf)[[code]](https://github.com/ppingyes/DIG-Net)

Deep Intensity Guidance Based Compression Artifacts Reduction for Depth Map

---
#Abstract

In this paper, we propose an deep intensity guidance based compression artifacts reduction model (denoted as DIG-Net) for JPEGcompressed depth map. The proposed DIG-Net model can learn an end-to-end mapping from the color image and distorted depthmap to the uncompressed depth map. To eliminate undesired artifacts such as discontinuities around object boundary, the proposedmodel is with three branches, which extracts the high frequency information from color image and depth maps as prior. Based onthe modified loss function with edge constraint, the deep multi-scale guidance information are learned and fused in the model tomake the edge of depth map sharper. Experimental results show the effectiveness and superiority of our proposed model comparedwith the state-of-the-art methods.
<p align="center">
<img src="https://i.imgur.com/o6NgOeE.png" width = 60% height = 30%/>
</p>
## Traning
  
	python DIG_net_train.py --quality=10
  
## Testing
  
	python DIG_net_test.py --quality=10 --IMAGE_NUM=1




# Citation

	@article{wang2018deep,  
 		 title={Deep Intensity Guidance Based Compression Artifacts Reduction for Depth Map},  
 		 author={Wang, Xu and Zhang, Pingping and Zhang, Yun and Ma, Lin and Kwong, Sam and Jiang, Jianmin},  
  		 journal={Journal of Visual Communication and Image Representation},  
  		 year={2018},  
  		 publisher={Elsevier}  
	}  
