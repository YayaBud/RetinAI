import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Microscope, ArrowRight, Lock, User, Eye, EyeOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { signInWithPopup, signInWithEmailAndPassword } from 'firebase/auth';
import { auth, googleProvider } from '../firebase';
import { useAuth } from '../context/AuthContext';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const { bypassAdmin } = useAuth();

  const handleBypass = () => {
    bypassAdmin();
    navigate('/app');
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    // Admin bypass: if operator username is admin, bypass firebase
    if (username.toLowerCase() === 'admin' || username.toLowerCase() === 'admin@retinai.local') {
      bypassAdmin();
      navigate('/app');
      return;
    }

    try {
      // For email/password login, we'll assume the username is an email for now
      // or map it if it's a legacy system.
      const email = username.includes('@') ? username : `${username}@retinai.local`;
      await signInWithEmailAndPassword(auth, email, password);
      navigate('/app');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Invalid username or password. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setIsLoading(true);
    setError('');
    try {
      await signInWithPopup(auth, googleProvider);
      navigate('/app');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Google sign-in failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#3b6a7a] flex items-center justify-center p-6 bg-[radial-gradient(circle_at_50%_50%,rgba(39,209,127,0.08)_0%,transparent_100%)] overflow-hidden relative">
      {/* Background Grid */}
      <div 
        className="absolute inset-0 opacity-10 pointer-events-none" 
        style={{
          backgroundImage: `linear-gradient(white 1px, transparent 1px), linear-gradient(90deg, white 1px, transparent 1px)`,
          backgroundSize: '40px 40px'
        }}
      />

      <div className="w-full max-w-md relative z-10">
        <div className="mb-8 text-center animate-in fade-in slide-in-from-bottom-5 duration-700">
          <div className="inline-flex items-center gap-2.5 p-3 rounded-2xl bg-[#2a5160] mb-6 shadow-xl border border-white/5">
            <Microscope className="w-8 h-8 text-[#27D17F]" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">Retina AI</h1>
          <p className="text-[#f0f5fa]/60">Clinical Intelligence Access</p>
        </div>

        <div className="bg-[#2a5160]/80 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border border-white/10 animate-in zoom-in-95 duration-500">
          {error && (
            <Alert variant="destructive" className="mb-6 bg-red-500/10 border-red-500/20 text-red-200">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.1em] text-[#27D17F]/80 ml-1">
                Operator Username
              </label>
              <div className="relative group">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30 group-focus-within:text-[#27D17F] transition-colors" />
                <Input
                  type="text"
                  placeholder="e.g. j.doe"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="bg-[#1e4250]/50 border-white/10 text-white pl-11 h-12 focus:ring-[#27D17F] focus:border-[#27D17F] placeholder:text-white/20 transition-all rounded-xl"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.1em] text-[#27D17F]/80 ml-1">
                Security Key
              </label>
              <div className="relative group">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/30 group-focus-within:text-[#27D17F] transition-colors" />
                <Input
                  type={showPassword ? 'text' : 'password'}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-[#1e4250]/50 border-white/10 text-white pl-11 h-12 focus:ring-[#27D17F] focus:border-[#27D17F] placeholder:text-white/20 transition-all rounded-xl"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/30 hover:text-white transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="pt-2">
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-12 bg-[#27D17F] hover:bg-[#22b86e] text-[#0a1e2c] font-bold rounded-xl shadow-lg shadow-[#27D17F]/20 group transition-all"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    Initiate Session
                    <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </Button>
            </div>
          </form>

          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-white/10"></span>
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-[#2a5160] px-3 text-white/30 font-medium tracking-[0.2em]">or cloud auth</span>
            </div>
          </div>

          <Button
            type="button"
            onClick={handleGoogleLogin}
            disabled={isLoading}
            variant="outline"
            className="w-full h-12 border-white/10 hover:bg-white/5 text-white bg-transparent rounded-xl"
          >
            <svg className="w-4 h-4 mr-3" viewBox="0 0 24 24">
              <path
                fill="currentColor"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="currentColor"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="currentColor"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="currentColor"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.66l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Sign in with Google
          </Button>

          <div className="mt-8 pt-8 border-t border-white/5 text-center">
            <p className="text-[10px] font-medium uppercase tracking-widest text-white/30 mb-4">
              Authorized Personnel Only
            </p>
            <div className="grid grid-cols-3 gap-2">
              <div 
                onClick={handleBypass}
                className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/5 flex flex-col items-center cursor-pointer hover:bg-white/10 transition-colors"
                title="Bypass Firebase as Admin"
              >
                <span className="text-[10px] text-white/40">ADMIN</span>
              </div>
              <div className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/5 flex flex-col items-center">
                <span className="text-[10px] text-white/40">DOCTOR</span>
              </div>
              <div className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/5 flex flex-col items-center">
                <span className="text-[10px] text-white/40">TECH</span>
              </div>
            </div>
          </div>
        </div>
        
        <p className="mt-8 text-center text-white/40 text-xs">
          Forgot credentials? Contact system administrator.
        </p>
      </div>
    </div>
  );
}
