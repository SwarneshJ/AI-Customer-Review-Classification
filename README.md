# AI-Customer-Review-Classification
A business-grade text classification system that transforms unstructured customer reviews into structured, decision-ready insights. Designed a theoretically grounded label schema identifying key customer issue types (delivery, product quality, billing/refund, service experience, and usability).

**Key contributions**:
• Created a detailed labeling codebook with definitions, examples, and edge-case rules
• Collected and cleaned 16,000+ reviews from public datasets (Amazon & Yelp)
• Benchmarked 6+ generative AI models (OpenAI, Anthropic, Meta, xAI) for labeling accuracy, cost, and reproducibility
• Fine-tuned a RoBERTa classifier using 15K labeled samples, achieving >0.80 macro-F1 on a human-labeled triple-annotated holdout set
• Conducted error analysis and built an evaluation framework with MCC, F1, AUC metrics
• Delivered an executive-style business memo outlining ROI, operational value, deployment strategy, and limitations

**Data Cleaning**

DataSet: https://www.kaggle.com/datasets/skamlo/food-delivery-apps-reviews?resource=download 

Data Cleaning: 

We decided to look for 1-star reviews to best understand the underlying reasons for poor app ratings for a food delivery company called GrubHub. 

We first applied high-level filters to isolate the relevant subset: 
 App = GrubHub 
 Rating = 1 star 
 Platform = App Store or Google Play 

That gave us around 50,000 reviews. 

Next, we used python to further clean for the following scenarios: 
Drop missing content 
Fix mojibake (‚Äô → ’ etc.) 
Normalize text (remove line breaks, collapse spaces) 
Remove very short reviews (Less than 20 characters) 
Drop duplicate reviews (Same username) 
Filter to English-only 

Save as a clean CSV for modeling. 

Then, we created two datasets. 15k reviews set, and 1k reviews for holdout 

