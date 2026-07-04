import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Shield } from "lucide-react";
import { loginUser, registerUser, fetchCurrentUser } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isRegister) {
        await registerUser(username, password);
      }

      const data = await loginUser(username, password);
      const user = await fetchCurrentUser(data.access_token);
      login(data.access_token, user);
      navigate("/history");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      className="min-h-screen flex items-center justify-center px-4 text-white"
      style={{ background: "linear-gradient(to bottom, #000000 0%, #140022 40%, #0a1f3f 100%)" }}
    >
      <div className="w-full max-w-md bg-neutral-900 rounded-2xl p-8 shadow-xl border border-white/10">
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-8 h-8 text-red-500" />
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Secure access</p>
            <h1 className="text-2xl font-bold">
              {isRegister ? "Create account" : "Sign in"}
            </h1>
          </div>
        </div>

        <p className="text-sm text-gray-400 mb-6">
          Detection is public. Sign in to view the spam call history log.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-lg bg-neutral-800 border border-white/10 px-4 py-3 outline-none focus:border-red-500"
              required
              minLength={3}
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg bg-neutral-800 border border-white/10 px-4 py-3 outline-none focus:border-red-500"
              required
              minLength={6}
            />
          </div>

          {error && (
            <p className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-red-600 hover:bg-red-700 disabled:opacity-60 text-white font-semibold py-3 rounded-lg transition-colors"
          >
            {loading ? "Please wait..." : isRegister ? "Register & sign in" : "Sign in"}
          </button>
        </form>

        <button
          type="button"
          onClick={() => {
            setIsRegister(!isRegister);
            setError("");
          }}
          className="w-full mt-4 text-sm text-gray-400 hover:text-white transition-colors"
        >
          {isRegister
            ? "Already have an account? Sign in"
            : "Need an account? Register"}
        </button>

        <Link
          to="/"
          className="block text-center mt-6 text-sm text-gray-500 hover:text-gray-300"
        >
          Back to detection
        </Link>
      </div>
    </section>
  );
}
