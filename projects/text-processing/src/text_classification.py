"""
Text Classification with Machine Learning
=========================================
This project demonstrates supervised learning using multiple classification algorithms
to classify text into topics (Technology, Science, Business, Sports, Health).

Author: Thura Win Kyaw
Date: 2025-11-19
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PHASE 1: DATASET CREATION
# ============================================================================

print("=" * 60)
print("PHASE 1: Creating Labeled Text Dataset")
print("=" * 60)

def create_text_dataset():
    """
    Create a synthetic labeled text dataset for topic classification.
    
    Returns:
    - DataFrame with text and category columns
    """
    # Define text samples for each category
    technology_texts = [
        "Artificial intelligence and machine learning are revolutionizing the tech industry. Deep learning algorithms can now process vast amounts of data and make predictions with remarkable accuracy. Neural networks have become the backbone of modern AI systems.",
        "Cloud computing has transformed how businesses store and access data. Companies are migrating to cloud platforms for better scalability and cost efficiency. Serverless architectures enable developers to build applications without managing infrastructure.",
        "Cybersecurity is crucial in protecting digital assets from threats. Encryption technologies safeguard sensitive information. Firewalls and intrusion detection systems prevent unauthorized access to networks.",
        "Blockchain technology offers decentralized and secure transaction systems. Cryptocurrencies use blockchain for transparent and immutable records. Smart contracts automate agreements without intermediaries.",
        "The Internet of Things connects everyday devices to the internet. Smart homes use IoT sensors for automation and energy efficiency. Industrial IoT improves manufacturing processes and predictive maintenance.",
        "Quantum computing promises to solve complex problems exponentially faster. Quantum algorithms can break traditional encryption methods. Research in quantum computing is advancing rapidly.",
        "Mobile app development has become essential for businesses. Native and cross-platform frameworks enable faster deployment. User experience design is critical for app success.",
        "Data science combines statistics and programming to extract insights. Big data analytics helps organizations make data-driven decisions. Data visualization makes complex information understandable.",
        "Software engineering practices ensure code quality and maintainability. Agile methodologies promote iterative development. Version control systems track code changes effectively.",
        "Virtual and augmented reality create immersive user experiences. VR headsets transport users to virtual worlds. AR overlays digital information onto the real world.",
        "Robotics and automation are transforming manufacturing industries. Industrial robots perform repetitive tasks with precision. Autonomous systems reduce human error and increase efficiency.",
        "5G networks provide ultra-fast internet connectivity. Low latency enables real-time applications. The Internet of Things benefits greatly from 5G technology.",
        "Edge computing processes data closer to the source. This reduces latency and bandwidth usage. IoT devices benefit from edge computing capabilities.",
        "DevOps practices bridge development and operations teams. Continuous integration automates testing and deployment. Containerization with Docker simplifies application deployment.",
        "Natural language processing enables computers to understand human language. Chatbots use NLP to interact with users. Language models can generate human-like text.",
        "Computer vision allows machines to interpret visual information. Image recognition systems identify objects in photos. Facial recognition technology has various applications.",
        "Software testing ensures applications work correctly. Automated testing reduces manual effort. Test-driven development writes tests before code implementation.",
        "Database management systems store and organize data efficiently. SQL databases use structured query language. NoSQL databases handle unstructured data effectively.",
        "Web development frameworks simplify building websites. Frontend frameworks create interactive user interfaces. Backend frameworks handle server-side logic.",
        "Open source software promotes collaboration and innovation. Developers contribute to projects freely. Open source licenses allow code sharing and modification."
    ]
    
    science_texts = [
        "The theory of evolution explains how species change over time. Natural selection favors traits that improve survival. Genetic mutations introduce variation in populations.",
        "Climate change is caused by increased greenhouse gas emissions. Global temperatures are rising due to human activities. Renewable energy can reduce carbon footprints.",
        "DNA contains genetic information passed from parents to offspring. Genes determine inherited characteristics. Genetic engineering can modify organisms for beneficial purposes.",
        "Photosynthesis converts sunlight into chemical energy in plants. Chlorophyll captures light energy. This process produces oxygen and glucose.",
        "The periodic table organizes chemical elements by atomic number. Elements combine to form compounds. Chemical reactions involve breaking and forming bonds.",
        "Newton's laws of motion describe how objects move. Gravity pulls objects toward Earth. Force equals mass times acceleration.",
        "Cells are the basic units of life. Organelles perform specific functions within cells. Cell division creates new cells for growth and repair.",
        "The water cycle describes how water moves through the environment. Evaporation turns liquid water into vapor. Precipitation returns water to Earth's surface.",
        "Electricity flows through conductive materials. Circuits provide paths for electric current. Voltage and resistance determine current flow.",
        "The solar system consists of planets orbiting the sun. Earth is the third planet from the sun. Gravity keeps planets in their orbits.",
        "Atoms are composed of protons, neutrons, and electrons. The nucleus contains protons and neutrons. Electrons orbit around the nucleus.",
        "Magnetism occurs when electric charges move. Magnets have north and south poles. Magnetic fields influence charged particles.",
        "Sound waves travel through air and other media. Frequency determines pitch. Amplitude determines volume.",
        "Light behaves as both waves and particles. Reflection and refraction change light direction. Colors result from different wavelengths.",
        "The human body contains multiple organ systems. The circulatory system transports blood. The nervous system controls body functions.",
        "Ecosystems include living organisms and their environment. Food chains show energy transfer. Biodiversity maintains ecosystem health.",
        "Plate tectonics explains continental movement. Earth's crust consists of moving plates. Earthquakes occur at plate boundaries.",
        "Stars form from clouds of gas and dust. Nuclear fusion powers stars. Stars eventually die and form new elements.",
        "Microscopes reveal structures too small to see. Electron microscopes provide high magnification. Microscopy advances scientific understanding.",
        "Scientific method involves observation and experimentation. Hypotheses are tested through experiments. Theories explain observed phenomena."
    ]
    
    business_texts = [
        "Marketing strategies help businesses reach target audiences. Digital marketing uses online platforms for promotion. Brand awareness increases customer recognition and loyalty.",
        "Financial planning ensures long-term business sustainability. Budgets allocate resources effectively. Cash flow management prevents liquidity problems.",
        "Leadership skills are essential for managing teams. Effective communication motivates employees. Decision-making requires analyzing risks and opportunities.",
        "Customer service builds strong relationships with clients. Satisfied customers become brand advocates. Problem resolution maintains customer satisfaction.",
        "Supply chain management optimizes product delivery. Logistics coordinates transportation and storage. Efficient supply chains reduce costs and delays.",
        "Entrepreneurship involves creating new business ventures. Innovation drives competitive advantage. Risk-taking is necessary for business growth.",
        "Human resources manage employee recruitment and development. Training programs improve workforce skills. Employee retention reduces turnover costs.",
        "Strategic planning sets long-term business goals. SWOT analysis evaluates strengths and weaknesses. Market research identifies customer needs.",
        "Sales techniques persuade customers to purchase products. Relationship building creates repeat customers. Negotiation skills close business deals.",
        "Project management ensures timely completion of tasks. Gantt charts visualize project timelines. Risk management prevents project failures.",
        "Accounting tracks financial transactions and performance. Balance sheets show assets and liabilities. Profit and loss statements reveal profitability.",
        "E-commerce enables online business transactions. Payment gateways process digital payments. Online stores reach global customers.",
        "Corporate culture influences employee behavior. Values and mission guide decision-making. Positive culture improves productivity.",
        "Investment strategies grow business capital. Diversification reduces financial risk. Long-term investments provide stable returns.",
        "Partnerships expand business capabilities. Joint ventures share resources and risks. Strategic alliances create competitive advantages.",
        "Quality control ensures product standards. Customer feedback improves quality. Continuous improvement enhances business processes.",
        "Business ethics guide moral decision-making. Corporate social responsibility benefits communities. Ethical practices build trust.",
        "Innovation drives business competitiveness. Research and development create new products. Technology adoption improves efficiency.",
        "Globalization expands business markets internationally. Cultural understanding prevents misunderstandings. International trade increases opportunities.",
        "Time management maximizes productivity. Prioritization focuses on important tasks. Delegation distributes workload effectively."
    ]
    
    sports_texts = [
        "Basketball requires teamwork and coordination. Players must shoot accurately and defend effectively. The game involves strategy and physical fitness.",
        "Football combines strength and strategy. Teams compete to score touchdowns. Training improves player performance and reduces injuries.",
        "Soccer is the world's most popular sport. Players use footwork and ball control. Teamwork is essential for winning matches.",
        "Tennis requires agility and precision. Players serve and return balls across the net. Mental focus is crucial during matches.",
        "Swimming improves cardiovascular fitness. Different strokes target various muscle groups. Competitive swimming requires technique and endurance.",
        "Running builds endurance and strength. Marathon training requires gradual distance increases. Proper form prevents running injuries.",
        "Baseball involves hitting and fielding skills. Pitchers throw different types of pitches. Strategy plays a key role in game outcomes.",
        "Golf requires precision and patience. Players aim for the lowest score. Course management is important for success.",
        "Olympic games showcase athletic excellence. Athletes train for years to compete. Sportsmanship and fair play are valued.",
        "Cycling improves leg strength and cardiovascular health. Road and mountain biking offer different challenges. Safety equipment prevents accidents.",
        "Volleyball requires jumping and spiking skills. Teams work together to score points. Communication is essential for coordination.",
        "Boxing combines strength and technique. Fighters train extensively for matches. Safety equipment protects athletes from injuries.",
        "Yoga improves flexibility and mental well-being. Different poses target various body parts. Meditation enhances mindfulness and relaxation.",
        "Weightlifting builds muscle strength. Progressive overload increases strength over time. Proper form prevents injuries.",
        "Skiing requires balance and coordination. Downhill and cross-country skiing differ in technique. Safety equipment is essential on slopes.",
        "Cricket is popular in many countries. Batsmen score runs by hitting the ball. Bowlers try to dismiss batsmen.",
        "Rugby is a physical contact sport. Teams compete to score tries. Strength and endurance are important.",
        "Gymnastics requires flexibility and strength. Athletes perform routines on various apparatus. Precision and artistry are valued.",
        "Martial arts teach self-defense and discipline. Different styles emphasize various techniques. Training improves physical and mental strength.",
        "Track and field includes running and jumping events. Athletes compete in various disciplines. Training improves speed and technique."
    ]
    
    health_texts = [
        "Exercise improves physical and mental health. Regular physical activity strengthens the heart. Exercise releases endorphins that boost mood.",
        "Nutrition provides essential nutrients for the body. Balanced diets include fruits and vegetables. Proper nutrition prevents chronic diseases.",
        "Sleep is crucial for physical and mental recovery. Adults need seven to nine hours of sleep. Sleep deprivation affects cognitive function.",
        "Mental health is as important as physical health. Stress management techniques reduce anxiety. Therapy helps people cope with challenges.",
        "Preventive care detects health problems early. Regular check-ups monitor overall health. Vaccinations prevent infectious diseases.",
        "Hydration is essential for body functions. Water regulates body temperature. Dehydration causes fatigue and health problems.",
        "Meditation reduces stress and improves focus. Mindfulness practices enhance mental well-being. Regular meditation improves emotional regulation.",
        "Cardiovascular health depends on exercise and diet. Heart disease is preventable through lifestyle changes. Regular exercise strengthens the cardiovascular system.",
        "Weight management requires balanced diet and exercise. Calorie balance determines weight changes. Sustainable habits maintain healthy weight.",
        "Dental health prevents oral diseases. Regular brushing and flossing remove plaque. Dental check-ups detect problems early.",
        "Vision care protects eye health. Regular eye exams detect vision problems. Protective eyewear prevents eye injuries.",
        "Bone health requires calcium and vitamin D. Weight-bearing exercises strengthen bones. Osteoporosis prevention starts early in life.",
        "Immune system function depends on lifestyle factors. Adequate sleep supports immune health. Stress weakens immune system function.",
        "Chronic disease management improves quality of life. Medication adherence is important. Lifestyle changes complement medical treatment.",
        "First aid knowledge saves lives in emergencies. CPR can restart stopped hearts. Basic first aid skills are valuable.",
        "Allergy management prevents severe reactions. Avoiding triggers reduces symptoms. Medications treat allergic reactions.",
        "Pain management improves quality of life. Various treatments address different pain types. Physical therapy helps manage chronic pain.",
        "Skin health requires protection from sun damage. Sunscreen prevents skin cancer. Regular skin checks detect problems early.",
        "Hearing health protects against noise damage. Ear protection prevents hearing loss. Regular hearing tests monitor function.",
        "Respiratory health depends on avoiding pollutants. Smoking damages lung function. Exercise improves respiratory capacity."
    ]
    
    # Combine all texts with labels
    all_texts = []
    all_labels = []
    
    categories = {
        'Technology': technology_texts,
        'Science': science_texts,
        'Business': business_texts,
        'Sports': sports_texts,
        'Health': health_texts
    }
    
    for category, texts in categories.items():
        all_texts.extend(texts)
        all_labels.extend([category] * len(texts))
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': all_texts,
        'category': all_labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Create dataset
df = create_text_dataset()

# Save dataset
os.makedirs('../text-processing/data', exist_ok=True)
df.to_csv('../text-processing/data/text_dataset.csv', index=False)
print(f"\n✅ Created dataset with {len(df)} samples")
print(f"✅ Dataset saved to '../text-processing/data/text_dataset.csv'")
print("\nCategory distribution:")
print(df['category'].value_counts())
print("\nFirst few samples:")
print(df.head())

# ============================================================================
# PHASE 2: TEXT PREPROCESSING
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 2: Text Preprocessing")
print("=" * 60)

def preprocess_text(text):
    """
    Clean and preprocess text.
    
    Parameters:
    - text: Input text string
    
    Returns:
    - Cleaned text string
    """
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

print(f"\n✅ Preprocessed {len(df)} text samples")
print(f"✅ Encoded {len(label_encoder.classes_)} categories: {list(label_encoder.classes_)}")

# ============================================================================
# PHASE 3: FEATURE EXTRACTION
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 3: Feature Extraction")
print("=" * 60)

# Prepare data
X = df['cleaned_text']
y = df['category_encoded']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# TF-IDF Vectorization
print("\n" + "-" * 60)
print("TF-IDF Vectorization")
print("-" * 60)
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,    # Minimum document frequency
    max_df=0.95  # Maximum document frequency
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ TF-IDF features: {X_train_tfidf.shape[1]}")

# Count Vectorization (for comparison)
print("\n" + "-" * 60)
print("Count Vectorization (Bag of Words)")
print("-" * 60)
count_vectorizer = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print(f"✅ Count features: {X_train_count.shape[1]}")

# ============================================================================
# PHASE 4: MODEL DEVELOPMENT
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 4: Model Development")
print("=" * 60)

# Use TF-IDF features (typically better than Count)
X_train_features = X_train_tfidf
X_test_features = X_test_tfidf

# Model 1: Multinomial Naive Bayes
print("\n" + "-" * 60)
print("Model 1: Multinomial Naive Bayes")
print("-" * 60)
model_nb = MultinomialNB(alpha=1.0)
model_nb.fit(X_train_features, y_train)
y_pred_nb = model_nb.predict(X_test_features)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy: {accuracy_nb:.4f}")

# Model 2: Logistic Regression
print("\n" + "-" * 60)
print("Model 2: Logistic Regression")
print("-" * 60)
model_lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model_lr.fit(X_train_features, y_train)
y_pred_lr = model_lr.predict(X_test_features)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.4f}")

# Model 3: Support Vector Machine
print("\n" + "-" * 60)
print("Model 3: Support Vector Machine (SVM)")
print("-" * 60)
model_svm = SVC(kernel='linear', random_state=42, C=1.0)
model_svm.fit(X_train_features, y_train)
y_pred_svm = model_svm.predict(X_test_features)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy: {accuracy_svm:.4f}")

# Hyperparameter tuning for best model
print("\n" + "-" * 60)
print("Hyperparameter Tuning (Logistic Regression)")
print("-" * 60)
param_grid = {
    'C': [0.1, 1, 10],
    'max_iter': [500, 1000]
}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_features, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_features)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test accuracy: {accuracy_best:.4f}")

# ============================================================================
# PHASE 5: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 5: Model Evaluation")
print("=" * 60)

# Evaluate all models
models = {
    'Naive Bayes': (model_nb, y_pred_nb),
    'Logistic Regression': (model_lr, y_pred_lr),
    'SVM': (model_svm, y_pred_svm),
    'Best Model (LR Tuned)': (best_model, y_pred_best)
}

results = []
for name, (model, y_pred) in models.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

results_df = pd.DataFrame(results)
print("\n" + "-" * 60)
print("Model Comparison Summary")
print("-" * 60)
print(results_df.to_string(index=False))

# Classification report for best model
print("\n" + "-" * 60)
print("Classification Report (Best Model)")
print("-" * 60)
print(classification_report(
    y_test, y_pred_best,
    target_names=label_encoder.classes_
))

# Cross-validation
print("\n" + "-" * 60)
print("Cross-Validation (5-fold)")
print("-" * 60)
cv_scores = cross_val_score(
    best_model, X_train_features, y_train,
    cv=5, scoring='accuracy'
)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# PHASE 6: VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 6: Visualization")
print("=" * 60)

os.makedirs('../text-processing/results', exist_ok=True)

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix - Text Classification', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category')
plt.ylabel('True Category')
plt.tight_layout()
plt.savefig('../text-processing/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

# 2. Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    values = [results_df.loc[i, metric] for i in range(len(results_df))]
    bars = ax.bar(results_df['Model'], values, color=['skyblue', 'lightgreen', 'coral', 'gold'])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('../text-processing/results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison.png")
plt.close()

# 3. Feature Importance (Top words per category)
print("\n" + "-" * 60)
print("Feature Importance Analysis")
print("-" * 60)

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# For each category, find most important features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Top Features (Words) per Category', fontsize=16, fontweight='bold')

for idx, category in enumerate(label_encoder.classes_):
    ax = axes[idx // 3, idx % 3]
    category_idx = label_encoder.transform([category])[0]
    
    # Get coefficients for this category
    coef = best_model.coef_[category_idx]
    
    # Get top 10 features
    top_indices = coef.argsort()[-10:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [coef[i] for i in top_indices]
    
    ax.barh(top_features, top_scores, color='steelblue')
    ax.set_title(f'{category}')
    ax.set_xlabel('Coefficient Value')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('../text-processing/results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: feature_importance.png")
plt.close()

# 4. Category Distribution
plt.figure(figsize=(10, 6))
category_counts = df['category'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='lightcoral')
plt.title('Category Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(category_counts.values):
    plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../text-processing/results/category_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: category_distribution.png")
plt.close()

# ============================================================================
# PHASE 7: MODEL PERSISTENCE & PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 7: Model Persistence & Predictions")
print("=" * 60)

# Save models
os.makedirs('../text-processing/models', exist_ok=True)
joblib.dump(best_model, '../text-processing/models/text_classifier.pkl')
joblib.dump(tfidf_vectorizer, '../text-processing/models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, '../text-processing/models/label_encoder.pkl')
print("✅ Saved: text_classifier.pkl")
print("✅ Saved: tfidf_vectorizer.pkl")
print("✅ Saved: label_encoder.pkl")

# Load and test saved model
print("\n" + "-" * 60)
print("Testing Saved Model")
print("-" * 60)
loaded_model = joblib.load('../text-processing/models/text_classifier.pkl')
loaded_vectorizer = joblib.load('../text-processing/models/tfidf_vectorizer.pkl')
loaded_encoder = joblib.load('../text-processing/models/label_encoder.pkl')

# Example predictions
print("\n📊 Example Predictions:")
print("-" * 60)
example_texts = [
    "Machine learning algorithms can process large datasets and make accurate predictions using neural networks.",
    "Photosynthesis converts sunlight into energy in plants through chlorophyll and produces oxygen.",
    "Marketing strategies help businesses reach customers through digital platforms and increase brand awareness.",
    "Basketball requires teamwork and coordination as players shoot accurately and defend effectively.",
    "Exercise improves physical health by strengthening the heart and releasing mood-boosting endorphins."
]

for text in example_texts:
    cleaned = preprocess_text(text)
    features = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(features)[0]
    category = loaded_encoder.inverse_transform([prediction])[0]
    probability = loaded_model.predict_proba(features)[0][prediction]
    print(f"\nText: {text[:60]}...")
    print(f"Predicted Category: {category} (Confidence: {probability:.2%})")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✅ Dataset: {len(df)} samples across {len(label_encoder.classes_)} categories")
print(f"✅ Models Trained: Naive Bayes, Logistic Regression, SVM")
print(f"✅ Best Model: Logistic Regression (Tuned) - Accuracy: {accuracy_best:.4f}")
print(f"✅ Visualizations: 4 plots saved to results/")
print(f"✅ Models Saved: Ready for deployment")
print("\n" + "=" * 60)
print("Project Enhancement Complete! 🎉")
print("=" * 60)
