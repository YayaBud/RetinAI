import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Microscope, ArrowRight, Lock, User, Eye, EyeOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch('http://localhost:8000/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Invalid credentials');
      }

      const data = await response.json();
      
      // Get user info
      const userResponse = await fetch('http://localhost:8000/users/me', {
        headers: {
          'Authorization': `Bearer ${data.access_token}`,
        },
      });
      const userData = await userResponse.json();

      login(data.access_token, { username: userData.username, role: userData.role });
      navigate('/app');
    } catch (err) {
      setError('Invalid username or password. Please try again.');
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

          <div className="mt-8 pt-8 border-t border-white/5 text-center">
            <p className="text-[10px] font-medium uppercase tracking-widest text-white/30 mb-4">
              Authorized Personnel Only
            </p>
            <div className="grid grid-cols-3 gap-2">
              <div className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/5 flex flex-col items-center">
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
