import React, { createContext, useContext, useState, useEffect } from 'react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import type { User as FirebaseUser } from 'firebase/auth';
import { auth } from '../firebase';

interface User {
  username: string;
  email: string;
  role: 'doctor' | 'technician' | 'admin';
  photoURL?: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  logout: () => void;
  isAuthenticated: boolean;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser: FirebaseUser | null) => {
      if (firebaseUser) {
        const idToken = await firebaseUser.getIdToken();
        const idTokenResult = await firebaseUser.getIdTokenResult();
        setToken(idToken);
        
        let role = idTokenResult.claims.role as string;
        if (!role) {
            role = firebaseUser.email === 'adityasa2838@gmail.com' ? 'admin' : 'doctor';
        }
        
        setUser({
          username: firebaseUser.displayName || firebaseUser.email?.split('@')[0] || 'User',
          email: firebaseUser.email || '',
          role: role as 'doctor' | 'technician' | 'admin',
          photoURL: firebaseUser.photoURL || undefined
        });
        
        localStorage.setItem('retinai_token', idToken);
      } else {
        setToken(null);
        setUser(null);
        localStorage.removeItem('retinai_token');
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const logout = () => {
    signOut(auth);
  };

  return (
    <AuthContext.Provider value={{ user, token, logout, isAuthenticated: !!token, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
