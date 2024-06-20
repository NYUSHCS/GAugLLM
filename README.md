# GAugLLM
Official code for "[GAugLLM: Improving Graph Contrastive Learning for
Text-Attributed Graphs with Large Language Models](https://arxiv.org/abs/2406.11945)". GAugLLM is a novel framework for augmenting TAGs, leveraging advanced large language models like Mistral to enhance self-supervised graph learning.

## Pipeline of the GAugLLM
The learning paradigm of GAugLLM vs. traditional GCL methods on TAGs. While standard GCL methodologies rely on text attributes primarily to generate numerical node features via shallow embedding models, such as word2vec, our GAugLLM endeavors to advance contrastive learning on graphs through advanced LLMs. This includes the direct perturbation of raw text attributes for feature augmentation, facilitated by a novel mixture-of-prompt experts technique. Additionally, GAugLLM harnesses both structural and textual commonalities to effectively perturb edges deemed most spurious or likely to be connected, thereby enhancing structure augmentation.

![img](https://github.com/NYUSHCS/GAugLLM/blob/main/img/pipeline.png)

## 🚀Quick Start
For Mix-of-Experts-Prompt part, please check LLMs folder. First you should follow GIANT:Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction (https://arxiv.org/abs/2102.06514) to set up and update our modifications. 
![architecture](https://github.com/NYUSHCS/GAugLLM/blob/main/img/moep.png)

For Collaborative Edge Modifier part, we add some modification on the original GCL frameworks of BGRL, GBT, GraphMAE and S2GAE. For GraphCL we used its loss function design in GBT framework. 

### Mix-of-Expert-Prompt
You can use run.sh in LLM folder to run Mix-of-Expert-Prompt over GIANT framework. 

### Collaborative Edge Modifier
To run original and GAugLLM on GCLs respectively, please use sripts with original or GAug in their names respectively. 
