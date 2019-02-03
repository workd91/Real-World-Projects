# Real-World-Projects
Degrees have become rather costlier so finding an alternative means to gain skill and knowledge for a job by solving real world problem. This isn't restricted to finding a job and this also help people to progress their career. Businesses want to mitigate risk so when they are hiring someone, they would know the behavior pattern of the person. Degrees from validate the knowledge and discipline of the candidate. 
Let's reverse engineer the logic of hiring (business needs) - Degrees signify lot of information that companies look for in finding real talent. At the end of the day businesses require people who deliver and can be an asset in progressing the business. They would like candidates to take responsibility and challenging organization roles. Businesses gain competitive advantage by having talented pool(Diverse cognitive skillset) of people in the organization. Real world problems are daunting than college projects and require lot of skillset to be gained while engineering a solution to the project. 
This repo is a guide to setting up a real world project that can enable people to gain knowledge on the problem and find solution. End of goal is to allow people to get hired as well progress their career. It's an alternative to college degree that can stay on the resume. 

Internet has created ton of information across domains and lot of free resources to learn. In order to be hired people have to unique. Most college degrees have same syllabus for everyone but in a market place we need people with unique skillset to resolve problem. Unique skillset are gained through pursual of real project in their own unique ways. 

One of the main concerns across sectors is the lack of talent due to changing technology. To stay relevant people have to be unique. 

Here are some of the real world problem that require solution in the machine learning space (organizations are also looking for solution): 

1. Document Understanding And Inference. 
2. Retail Supply Chain & Inventory Forecasting
3. Image Enhancement (Super-Resolution)

Machine Learning has grown by leaps and bounds. Technology enables resolution of existing problems with new approaches.
Machine Learning, Computer vision and deep learning projects are not only limited to coding. There is a whole array of process that preceeds and super-seceeds them. 

There are so many aspects to a problem. For example finding data pairs for GAN's is a difficult task. Image enhancement and super-resolution of images is an important problem the graphics and image processing industries is facing. Most of the deep learning solution use downsampled data that doesn't model real world noise or distribution (Bicubic or Bilinear Interpolation is used for downsizing and generating a pairs of low resolution and high resolution). One solution to the problem of data pair creation is to use a unpaired domain-domain translation GAN to transfer noises and distribution of low resolution images from other domain. Finding domain specific data is hard. This enables existing neural networks to learn new features about reconstructing images from noisy low-resolution images. 

Likewise there could be numerous problem that are require solution. Companies are always looking for talents who could solve real world problem since the business revolves around it. 

<H1>Document Understanding And Inference</H1>: 
    Business operations revolve around inferring scenarios and finding solution to the problem. Most business spend so much time in processing data to find insight that can help their clients and customers. Financial and Tactical industries rely on understanding techniques that work in the marketplace. Annual reports created by organization carry lot of potential information. It showcase their intent, area of interest, methods used to productionize and market the product, revenue streams, dissection of the customers, their current short coming, potential look out for solution, channel of sale, marketing platforms utilized, revenue vs cost incurred, comparison with the previous yearly or quarterly performance, etc.
    
This type of inference requires multi-step logical understanding of the context. It's a laborious process to evaluate each annual reports to identify key metrics. Natural language model have made great progress in 2018 with the advancement of language modelling that enables contextual understanding of the text. 
The problem is broken down into 
1. Problem Recognition
2. Present Solution In The Market
3. Data Collection With Variance But Without Bias
4. Data Pipeline For Data Injection 
5. Neural Architecture - State Of The Art Networks - Case Study
6. Engineering A Network - Mathematical Modelling, Networks To Use, Loss Function, Hyperparameter etc. 
7. Data Evaluation - Metrics
8. Out Of Distribution Data
9. Edge Case Analysis
10.Model Production 

11.Performance Evaluation In Real World - Testing In Diverse Variance In The Real World And Setting Benchmarks

12.What Business Value Does The Solution Provide? 

Each part of the problem can itself be a task that businesses are after. I will share as much resource of information to make the path to solution easier but it's a hard task. This will differentiate your skillset. 

Here are some of the resources to get your started 

Problem recognition steps for this problem to understand the business requirement. Financial institutions, corporate business have enough data but they need contextual model to convert unstructured data into a business insight. www.cio.com provides survey and insights from the CIO across multiple domain. They showcase what are the painpoints organization are facing at the moment. Case study is always a first step to understanding the problem. 

At Present most of the solution utlizies word embedding (word meaning based on nearby words occuring together). Most of the words carry different meaning depending on the context where they occur (Polysemy). Identifying relationship between words and sentences in which they occur gives semantic and linguistic relationship.

Machine learning models are good at interpolating. Data with good amount of variance is required. 

For datasets -- www.annualreports.com

For Language Modelling - Contextual Neural Network - Elmo - > https://github.com/allenai/bilm-tf

Bert (Language Modelling Network)--> https://github.com/google-research/bert


<h1> Image Enhancement </h1>
Image enhancement has lot of potential use cases from computer vision problem like self driving cars, retail self checkouts, aerial/ satellite imaging, drone imaging, and many other. Image enhancement is basically upsampling the image size by understanding how to reconstruct high resolution images from a given image. Image captured from mobiles and camera have noise. The resolution capacity varies across different camera. Lighting variation in the real world impact the amount of information captured by the camera sensors. 

When we segregate noise there are
<li> Shot Noise </li>
<li>Guassian Noise </li>
<li>Salt And Pepper Noise </li>
<li>Sensory noise </li> 
<li>Motion Blur </li>

Image enhancement methods need to understand ability to reconstruct image pixel from noisy pixel and low light pixels. Images contain soft boundary transistion between foreground and background objects. Reconstructing image pixel is a difficult task considering how real world physics impacts the images. Physics is everywhere so there are lot of variance in the data due to material property of the object like reflection, illumination etc. These stochasticity make the task harder to reconstruct original pixels. Earlier approaches used nearby pixels technique to reconstruct the pixels that aren't good. Deep learning took a step further to identify interpolate pixels from the data that it's trained on. Deep learning are coming due to their inability to generate proper low resolution and high resolution pairs of images for training. Modelling the physics variable for low-resolution image is difficult so most of the deep learning model use bicubic downsampling or bilinear downsampling to generate low resolution images. These low resolution images aren't modelling real world noises. 

Let's breakdown what happen when you use a bicubic downsampling? You are basically sampling nearby pixels into a single pixel. Number of nearby pixel to consider depends on the downsampling proportion. The task of the neural network is to predict nearby pixels from the information from single pixel. Downsampling can cause aliasing of image pixel. Aliasing is a point beyond which pixel doesn't carry any descriptive detail about the image. Model learns to retrieve information from pixels that may carry partial information to that of the high resolution images. Neural network learn higher level features like object shapes and their correlation with the background. That's why Generative Adversarial Network (GAN) are more reasoning models. 

Breaking down the problem into smaller chunks 
<li> Noise in Images </li>
<li> Data Pair Generation </li>
<li> Neural Architecture </li>
<li> Multi-task Optimization </li> 
<li> Loss Function </li>
<li> High Imag Resolution - The Hardware Problem </li>
<li> Use Case </li>
<li> Industries Looking Forward To The Solution </li>

<h1> Data Pair Generation </h1> 
Data pair generation for generative model is a challenging task since there is lack of domain specific while data may be spread out in other domain. Image enhancement will most likely be domain specific for example if you're enhancing human images then it's confined to it. Physics variation that can occur in the real world should have to be addressed in the data collection/creation pipeline. When it comes to computer vision problem there are so many aspects in an image to consider. One such thing is light variation and object in the scene illuminate in a different fashion depending on their material property. It can impact the pixels captured in the image. It's difficult to model all the physics variation using a augmentation. One option try is to use unpaired domain translation GAN that can transfer noise and lighting variation other domain to your image. In order to make the image enhancement work we can build a high-resolution to low res gan that's unpaired to transfer noises from a different domain. To a certain extent this alleviate the issue with respect to having a data-pair. 

Here are some of resources to peek your curious minds into 
<h2> Multi-task Optimization </h2>
    Neural networks learn depending on the target task. For a good image enhancement model, neural network should be to handle pixel level information retrieval as well as high level information. High level information being object features. A classification task combined to a image enhancement model can provide high level information about the object shape and texture. It's important to have both local and global context. 
<h2> Loss Function </h2>
The loss function are crucial to generative network as they shape the outcome. Proper design of loss function for the generator network & discriminative network aids faster training time. Discriminator network is the one that helps feedback crucial information back to the generator. Perceptual loss carries high level information about the scene while a wasterstein loss or jensen loss. 

<h2> The High Resolution- The Hardware Problem </h2>
Processing bigger image resolution could be challenging if there is GPU constrain since even the neural network would prefer a downsized data with smaller pixel values. Reconstructing a large resolution image could be difficult so we can reduce the problem we are trying to solve by using neural network to create filter that when multiplied/concatenated with the target image gives high resolution look. Basically you can upsample the input image using bicubic or bilinear interpolation and concatenate with the output of the network before passing it through the loss function. This way you reduce the problem to creating a network that does a smaller task and as well we have overcome the boundary of processing large volume of images. 

<h2> Industrial Use Care </h2>
The graphic and gaming are predominantly involved in image processing have shown interest in image enhancement. Lighting variation can cause noisyness in the image. It's extremely time consuming to develop good designs. Deeplearning models are data hungry and having a high-resolution images creates models with better accuracy and lower training times. In order to overcome domain shift that can happen during inference,  image enhancement models can be deployed. Since we can transfer the domain variation from one domain to other we are able to create a better enhancement model so that deep learning neural networks don't have to deal with such physics variation. Most of the classification models are task specific so they don't have global context and relationship with other objects in the image. While GAN's have to understand the overall scene relation so the output from GAN's are good for neural network that perform downstream task. Most of the computer vision models don't have control over the low level pixel so they have to learn all the variation in the input without overall context. So pixel variation due to lighting, reflectance and illumination can affect accuracy of the model. 

Having a image enhancement prior to the downstream computer vision task can be great performance enhancer. 




