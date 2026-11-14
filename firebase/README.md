# Firebase setup
1. console.firebase.google.com -> Create project.
2. Build -> Firestore Database -> Create database (start in production mode).
3. (Optional, needs Blaze plan) Build -> Storage -> Get started. Copy the bucket name.
4. Project settings -> Service accounts -> Generate new private key.
   Save the file as `firebase/serviceAccountKey.json`  (NEVER commit or share it).
5. In config.json set: "firebase": {"enabled": true, "storage_bucket": "<your bucket>", ...}
   - Free Spark plan (no Storage): set "upload_screenshots": false.
6. `pip install firebase-admin` then run `python main.py`.
