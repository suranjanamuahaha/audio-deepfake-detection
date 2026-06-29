import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Clock, LogOut } from "lucide-react";
import { fetchSpamHistory } from "../api/client";
import { useAuth } from "../context/AuthContext";

function formatDetectedAt(isoString) {
  const date = new Date(isoString);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function SpamHistory() {
  const { token, user, logout } = useAuth();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchSpamHistory(token)
      .then((data) => {
        setItems(data.items || []);
        setTotal(data.total || 0);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [token]);

  return (
    <section
      className="min-h-screen px-4 pt-28 pb-16 text-white"
      style={{ background: "linear-gradient(to bottom, #000000 0%, #140022 40%, #0a1f3f 100%)" }}
    >
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-500">Protected log</p>
            <h1 className="text-3xl md:text-4xl font-bold mt-2">Spam Call History</h1>
            <p className="text-sm text-gray-400 mt-2">
              Signed in as <span className="text-white font-mono">{user?.username}</span>
            </p>
          </div>

          <div className="flex gap-3">
            <Link
              to="/"
              className="px-4 py-2 rounded-lg border border-white/10 hover:bg-white/5 transition-colors"
            >
              Back to detection
            </Link>
            <button
              onClick={logout}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>

        <div className="bg-neutral-900 rounded-2xl border border-red-500/20 overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              <span className="font-semibold">Detected spam / deepfake calls</span>
            </div>
            <span className="text-sm text-gray-400">{total} total</span>
          </div>

          {loading && (
            <div className="px-6 py-12 text-center text-gray-400">Loading history...</div>
          )}

          {error && (
            <div className="px-6 py-12 text-center text-red-400">{error}</div>
          )}

          {!loading && !error && items.length === 0 && (
            <div className="px-6 py-12 text-center text-gray-400">
              No spam calls logged yet. Run a detection on the home page — spam results are saved automatically.
            </div>
          )}

          {!loading && !error && items.length > 0 && (
            <div className="divide-y divide-white/5">
              {items.map((item) => (
                <div
                  key={item.id}
                  className="px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3"
                >
                  <div>
                    <p className="font-semibold text-red-400 capitalize">{item.label}</p>
                    <p className="text-sm text-gray-400 mt-1">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </p>
                  </div>

                  <div className="flex items-center gap-2 text-sm text-gray-300">
                    <Clock className="w-4 h-4 text-gray-500" />
                    <span>{formatDetectedAt(item.detected_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
