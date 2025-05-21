# SCAPE
The open source code for the ACM Web Conference 2025 (WWW ’25) paper "Panoramic Interests: Stylistic-Content Aware Personalized Headline Generation".
The PyTorch implementation source code will be ready soon.


![intro_new_v2](https://github.com/user-attachments/assets/b44bad96-4e54-41ed-8c66-6a9bd929e0cd)
**Figure 1. Illustration of the Joint Influence of Content Interests and Stylistic Preferences on Headline Personalization.**


Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work.
```
@inproceedings{lian2025space,
  title={Panoramic Interests: Stylistic-Content Aware Personalized Headline Generation},
  author={Junhong Lian and Xiang Ao and Xinyu Liu and Yang Liu and Qing He},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  year={2025}
}
```

## Abstract
Personalized news headline generation aims to provide users with attention-grabbing headlines that are tailored to their preferences. Prevailing methods focus on user-oriented content preferences, but most of them overlook the fact that diverse stylistic preferences are integral to users’ panoramic interests, leading to suboptimal personalization. In view of this, we propose a novel **S**tylistic-**C**ontent **A**ware **Pe**rsonalized Headline Generation (SCAPE) framework. SCAPE extracts both content and stylistic features from headlines with the aid of large language model (LLM) collaboration. It further adaptively integrates users’ long- and short-term interests through a contrastive learning-based hierarchical fusion network. By incorporating the panoramic interests into the headline generator, SCAPE reflects users’ stylistic-content preferences during the generation process. Extensive experiments on the real-world dataset PENS demonstrate the superiority of SCAPE over baselines.


## The Framework of SCAPE
![model_new_v4](https://github.com/user-attachments/assets/59e8ec43-af8d-4385-8004-e8fbca9d9415)


## Requirements
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```

## Update
[2025-05-21]  We would like to acknowledge that the comments in this project have been automatically regenerated using `Gemini-2.5-Flash`.