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

Having a image enhancement prior to the downstream computer vision task can be great performance enhancer. Let's break down what it can learn it's not as simple as smoothening out the edges of the blurry image. It has to understand the geometry of the object & their relation in the scene to enhance the resolution of the image. The soft transistion of image texture has to be understand by the model so that the edges are sharp. While enhancing the resolution of the image the information contained in one pixel has to be recreated in the each neighboring pixel instead of smoothening the pixel value. Smoothening often results in blurry edges. So the image enhancement model require local pixel level information as well as global context of the object. That's why it's often beneficial to joint optimize task so that one task can learn from other. Coupling object detection/classification with image enhancement can allow the model to learn faster and object level features. When it comes to semantic segmentation task we need to precisely understand the boundary transistion of the objects in the image. Semantic segmentation helps to understand the whole scene. Supervised learning task speed up the training process in the image enhancement GAN's. 

<li> Graphic Designing/ 3D Designing </li>
<li> Gaming </li>
<li> Computer vision Task (Consumer Facing Apps/ Industrial Apps) </li>
<li> Digital Marketing </li> 
<li> Social Media Content Creation </li>
<li> AR </li>
<li> Movies </li> 

<h1> Retail Operation </h1> 
After a fierce competition from ecommerce stores, retail stores have adopted new technology to enhance their existing line of business. Most of the retail decisions in merchandising, sourcing, store design, sales, personalization are based on information stored in spreadsheets. As the world is transistioning from spreadsheets to better analytical tools so that employees can have access to tools and insights for intelligent decision making. Entire supply chain decisions are made from processing different forms of data. Metrics that were used prior to account for different variations of data like images, text, and numerical were biased, lacked contextual information, lacked proper metric to evaluate asthetics of the image, consumer reaction etc. 

One of the major concerns with retail stores and ecommerce stores is the lack of skilled labor due to changing technologies. Organizations are unable to adopt new technology because they lack the expertise to setup a proper employee transistion corporate training program that would provide tools and insight that allow professional to progress their domain expertise to work different levels of information to create values.   

Some of the problems that retail stores face 
<li> Demand Estimation </li>
<li> Inventory Management </li>
<li> Last Mile Delivery Challenge </li>
<li> Supply Chain Optimization </li>
<li> Merchandising </li>
<li> Staffing </li>

<h2> Demand Estimation </h2> 
The rise of e-commerce stores have put a pressure on retail stores to adapt innovative strategies. The online shopping gave rise to a same day purchase patterns. People's shopping patterns are influenced by the videos and trends happening on the social media. Urban cities are a challenge for retail stores due to high-frequency & low volume of orders. This randomness in purchase behavior has created uncertainty in supply chain and demand estimation. Retail stores have becomes hubs for faster delivery in cities so replenishing inventory is a challenge due to traffic and parking concerns. Massive survey among CIO's of the organization showcased that one of the main concerns were demand estimation. Lack of inventory could cost them ton of sales. Walmart in the third quarter of 2018 hit with a inventory problem in the retail stores. Target also felt the impact of lack of inventory with orders to fulfil. How do we evaluate demand for product that different lifecycle in the retail space. 

Proper demand estimation could reduce 
<li> Constant replenishment to the retail stores </li>
<li> Transportation cost </li>
<li> Employees can focus on customer </li>
<li> Lack of inventory as well as inventory pile up </li>

It would also mean equipping the staff at the store with tools and insight that can enable them to make multi-step reasoned out decision. Most of the operation in retail space happen in the spreadsheets. Omnichannel style of having all the retail activity in one place can give a overview of the scene. It requires well-reasoned out analysis to evaluate the factors that influenced the sale by considering events that led to the scenario. 

One of the difficult challenges is combining data that are spreadout in different forms like images, text, number etc. Previous held out standard of analysis would convert data into numerical features to evaluate this would mean the contextual and visual aesthetics of the data is lost. These data are important as it impact how customers make decision everyday looking at the shelves where the products are placed. Forecasting demand is challenging task since the variation in data can be from many different factors like external climate, gasoline prices, inflation, job/wage growth, interest rates, stock market prices, consumer sentiment, housing prices, taxes, private investments, consumption vs saving, growth capital availability etc. 

Some products have smaller lifecycle while some product have longer lifecycle and it's particularly difficult to perfect piece of data that can represent the customer behavior patterns. There are many machine learning technique that provide good result in estimating demand. We have seen uber use a demand estimation model to predict the number of rides that would happen in a given day. 

Stock availability is a key variable in retaining the customer and generating a dense volume of order from them. Especially in the ecommerce filled world it's difficult to retain customers as they can find better offers in other places. Prediction of the demand for retail product would allow stores to structure their operation in a deterministic way thereby accomplishing cost reduction and optimal usage of resources. Point of sale (POS) data isn't enough to figure out the events that influenced customers to buy today vs tomorrow. But most of POS data isn't represented in the form that they naturally occur. That's why we need model that can process images (color, texture, scenes etc), contextual nature of text present in those packaging, brand popularity, brand contents & their impact etc. 

Deep Learning offers good solutions to model different variation in data. We could combine image features using Convolution networks with contextual nlp models that recognize contextual meaning of the sentence. Representing data in higher dimension can help identify trends, factors that influence the behavior pattern of the consumer. 

Real Time Sales data is available on amazon.com where they publish top 100 selling products each hour. You can build a dataset that would represent features of the products and other features associated with the product. You can create sales models that can show factors that impact sales and influence consumer behavior. 

Data Source - www.amazon.com & while kaggle and other data science community has lot of data about sales prediction. A real time dataset would be more ideal. amazon.com would be ideal space to generate real time data and see how your model fairs in real time. 

Simple Regression and Prediction models to start the transition to more realistic models 
Image Convolution Reads And Repos to get started - 
Contextual NLP Models -- 
Some of the retail space breakdown to learn more about the problem. Domain specific knowledge is helpful in correlating different factors that can impact the customer behavior patterns. Cognitive skills coupled with domain expertise brings better talent pool to organization. 

<h1> Inventory Management </h1> 
Inventory management is a most crucial aspect of retail business cycle. Proper management of inventory would enable faster fulfilment and reduce build up of inventory that could derail the momentum of doing business. Retail are innovating at every front to reach customers. Social media has given rise to brand content that interact with the customers. Customers prefer to know more about the organization they buy the product from. The prefer value content over marketing material and customer prefer brands that coincide with their values. Retail stores source products based on multiple factors to evaluate how well the product would do when placed in the shelves of the store. 

From previous data retailers could create models on price points, brand popularity, social media following, other regional/ecommerce sale of similar products, other competitors, etc while sourcing new products. At the end of the day they would want the product to sell-through. Shelf spaces are becoming costlier as the retail operation become costlier due to low volume & high frequency orders. Getting a sell-through is important for the retailer as well as the seller. Monitoring their reach on social media can help track the volume of products sold. Sellers task doesn't end just having the product placed at the shelf. They would need constant interaction with the customer through value contents to become a brand. Inventory management does mean managing inventories, warehouses, replenishment, logistics, and demand estimation. On-time delivery of products to central hub would be key to reducing the supply chain cost. Retailers sometime work with multiple logistic vendors to fulfil the requirement. Lot of technology has to invested to monitor real world traffic and climate to evaluate the time to the hubs. IoT solution are deloyed delivery traffic to find real time traffic. Many retailers are investing in automating the warehouses to fulfil orders faster. Automated warehouse still require high level engineering to make the process seamless. Warehouse management would become finding optimal conveyors, reducing maintenance & electricity cost, retrainable robotic arms for automated packaging, space allocation based on number of orders on the product, shorter retrieval time etc. Inventory management would require engineers, analysts, and management executive to make the process smooth. 

Supplychain can be affected by differential factors it requires forecasting models to anticipate any delay in shipment, alternative to setup to fulfil nearby fulfilment centeres. Most of the time retailers stick with one vendor but sometime they may have to utilize others to meet the demand in holiday season. At the end of the day the cost have to be managed efficiently. Some retailers even charge vendors for failure to deliver atleast 95% of goods on time. Big box retailers usually stock up for a year since they would to expand the operation. It could be a time lag for the sellers as their inventories are held up and they are paid for 90 days of sale. Lot of challenge to figure proper solution in the inventory management space. 
