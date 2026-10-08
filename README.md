# 🌸 Iris Flower Species Classifier (Vercel-Ready)

A modern, serverless machine learning web application that predicts Iris flower species (**Iris Setosa**, **Iris Versicolor**, and **Iris Virginica**) using a pre-trained scikit-learn model (`iris_model.pkl`). Specifically architected for seamless deployment on **Vercel** with a Python Serverless API and static CDN frontend.

🔗 **GitHub Repository:** [https://github.com/ekraislam/iris-flower-classifier.git](https://github.com/ekraislam/iris-flower-classifier.git)

---

## 📌 1. Project Overview

This project provides an end-to-end classification system for the famous Fisher's Iris dataset:
- **Serverless Backend:** Python serverless function (`api/index.py`) hosted on Vercel, eliminating the need for a persistent 24/7 server.
- **Modern Frontend:** Fast, responsive UI (`public/index.html`, `public/style.css`, `public/script.js`) with 1-click test presets, real-time input validation, loading spinners, and confidence metrics.
- **Model Inclusivity:** Directly loads and performs inference on the pre-trained `iris_model.pkl` without retraining.

---

## 🔬 2. How the Model Was Trained

The included model (`iris_model.pkl`) is based on the canonical **Iris Flower Dataset**:
- **Dataset Properties:** 150 flower samples across 3 balanced species classes (50 samples each of *Setosa*, *Versicolor*, and *Virginica*).
- **Features Evaluated:**
  1. `sepal_length` (cm) — Sepal length
  2. `sepal_width` (cm) — Sepal width
  3. `petal_length` (cm) — Petal length
  4. `petal_width` (cm) — Petal width
- **Algorithm:** Scikit-Learn **Random Forest Classifier** (`RandomForestClassifier`), an ensemble of decision trees that outputs the majority voting class for robust generalization.
- **Target Classes:**
  - Class `0` ➔ **Iris Setosa** (🌸 small petals, arctic hardy)
  - Class `1` ➔ **Iris Versicolor** (🌿 medium petals, blue flag)
  - Class `2` ➔ **Iris Virginica** (🌺 large petals, deep violet)
- **Serialization:** Exported via `joblib.dump()` as `iris_model.pkl`.

---

## ⚙️ 3. How the Web Application Works

```text
┌─────────────────────────────────┐
│     Client Browser / Mobile     │
│  (index.html, style.css, .js)   │
└────────────────┬────────────────┘
                 │ 1. POST /api/predict (JSON payload)
                 ▼
┌─────────────────────────────────┐
│     Vercel Serverless Edge      │
│          (vercel.json)          │
└────────────────┬────────────────┘
                 │ 2. Routes to Python Function
                 ▼
┌─────────────────────────────────┐
│          api/index.py           │
│  - Validates numeric inputs     │
│  - Checks positive dimensions   │
│  - Loads iris_model.pkl (joblib)│
│  - Runs model.predict([[...]])  │
│  - Computes probabilities       │
└────────────────┬────────────────┘
                 │ 3. Returns JSON Response
                 ▼
┌─────────────────────────────────┐
│  Live Result Card Updated       │
│  - Species: Setosa/Versi/Virgi  │
│  - Confidence: e.g. 100%        │
│  - Probabilities breakdown      │
└─────────────────────────────────┘
```

1. **User Input:** The user fills in the four measurements or selects one of the 1-click test presets (**Setosa**, **Versicolor**, or **Virginica**).
2. **API Request:** When the user clicks **Predict Species**, the frontend sends a `POST` request to `/api/predict` with JSON data:
   ```json
   {
     "sepal_length": 5.1,
     "sepal_width": 3.5,
     "petal_length": 1.4,
     "petal_width": 0.2
   }
   ```
3. **Serverless Inference:** The Python handler in `api/index.py` formats the features into a 2D numpy array `[[5.1, 3.5, 1.4, 0.2]]` and invokes `model.predict()`.
4. **Instant Response:** The API returns the predicted species, botanical profile, confidence score, and probability distribution.

---

## 📁 4. Project Directory Structure

```text
iris-flower-classifier/
├── api/
│   └── index.py            # Vercel serverless Python API endpoint
├── public/
│   ├── index.html          # Static HTML web interface
│   ├── style.css           # Modern glassmorphism stylesheet & responsive design
│   └── script.js           # Client-side API caller & form logic
├── iris_model.pkl          # Pre-trained Random Forest model
├── requirements.txt        # Minimal Python dependencies for Vercel
├── vercel.json             # Vercel serverless routing & asset bundling config
├── .gitignore              # Git ignore rules for caches & virtual environments
└── README.md               # Documentation & deployment guide
```

---

## 💻 5. Local Development Instructions

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/ekraislam/iris-flower-classifier.git
cd iris-flower-classifier
```

### Step 2: Install Python Dependencies
Ensure you have **Python 3.10+** (Python 3.12 recommended):

```bash
pip install -r requirements.txt
```
*(Or on Windows with the Python launcher):*
```powershell
py -m pip install -r requirements.txt
```

### Step 3: Run the Application Locally
You can run the server directly using Python:

```bash
python api/index.py
```
*(Or on Windows: `py api/index.py`)*

Output:
```text
[OK] Successfully loaded model from: .../iris_model.pkl
[OK] Local development server running at: http://127.0.0.1:5000
```

### Step 4: Open in Your Browser
Visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

*(Alternatively, if you have the Vercel CLI installed, you can simply run `vercel dev` in the root folder).*

---

## 🚀 6. Vercel Deployment Instructions

Follow these exact steps to deploy directly from your GitHub repository onto Vercel:

### Option A: Deploy via Vercel Web Dashboard (Recommended)

1. **Push Your Changes to GitHub**:
   Ensure all files (`api/index.py`, `public/`, `vercel.json`, `requirements.txt`, `iris_model.pkl`) are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure project for Vercel serverless deployment"
   git push origin main
   ```

2. **Log In to Vercel**:
   Go to [https://vercel.com](https://vercel.com) and log in with your GitHub account.

3. **Import the Repository**:
   - On the Vercel dashboard, click **"Add New..."** ➔ **"Project"**.
   - Under *Import Git Repository*, find and select `iris-flower-classifier` (or paste `https://github.com/ekraislam/iris-flower-classifier.git`).
   - Click **Import**.

4. **Configure Project Settings**:
   - **Project Name:** `iris-flower-classifier` (or your preferred name).
   - **Framework Preset:** Select **"Other"** (Vercel will automatically read `vercel.json`).
   - **Root Directory:** `./` (Leave as default).
   - **Build and Output Settings:** Leave all fields at their default settings (no build command is required because the static files are in `public/` and the backend is serverless Python).
   - **Environment Variables:** None required.

5. **Deploy**:
   - Click the **"Deploy"** button.
   - Vercel will install the dependencies from `requirements.txt`, compile the serverless function, and host your static files globally.
   - Within 1–2 minutes, you will receive your live production URL (e.g. `https://iris-flower-classifier.vercel.app`).

---

### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI** (requires Node.js):
   ```bash
   npm install -g vercel
   ```

2. **Deploy from Terminal**:
   Inside the project directory, run:
   ```bash
   vercel --prod
   ```

3. Follow the on-screen prompts to link your project and deploy.

---

## 🧪 7. API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/predict` | `POST` | Accepts JSON measurements, runs inference on `iris_model.pkl`, and returns classification. |
| `/api/health` | `GET` | Verifies serverless function uptime and model loading status. |
| `/` | `GET` | Serves the interactive static web application from `public/index.html`. |

#### Example Request:
```bash
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

#### Example Response:
```json
{
  "success": true,
  "result": {
    "species": "Setosa",
    "scientific_name": "Iris setosa",
    "confidence": 100.0,
    "probabilities": {
      "Setosa": 100.0,
      "Versicolor": 0.0,
      "Virginica": 0.0
    }
  }
}
```

---

## 👤 Author

- **Ekramul Islam (Ohi)**
- GitHub: [@ekraislam](https://github.com/ekraislam)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
Copyright &copy; 2026 Ekramul Islam (Ohi).
