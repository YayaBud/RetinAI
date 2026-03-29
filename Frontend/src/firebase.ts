import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyD6BDMiQfHf0O7caOvaOQ4x6G1XhyO5OcI",
  authDomain: "retinai-dashboard-102044.firebaseapp.com",
  projectId: "retinai-dashboard-102044",
  storageBucket: "retinai-dashboard-102044.firebasestorage.app",
  messagingSenderId: "138628225089",
  appId: "1:138628225089:web:61077f930c9c7fe07e8ec0"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
