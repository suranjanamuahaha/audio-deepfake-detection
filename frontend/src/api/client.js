const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function getApiUrl() {
  return API_URL.replace(/\/$/, "");
}

export async function predictAudio(blob) {
  const formData = new FormData();
  formData.append("file", blob, "audio.webm");

  const response = await fetch(`${getApiUrl()}/predict`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || data.error || "Prediction failed");
  }

  if (!data.success) {
    throw new Error(data.error || "Prediction failed");
  }

  return data;
}

export async function loginUser(username, password) {
  const response = await fetch(`${getApiUrl()}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Login failed");
  }

  return data;
}

export async function registerUser(username, password) {
  const response = await fetch(`${getApiUrl()}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Registration failed");
  }

  return data;
}

export async function fetchSpamHistory(token) {
  const response = await fetch(`${getApiUrl()}/history/spam`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Failed to load spam history");
  }

  return data;
}

export async function fetchCurrentUser(token) {
  const response = await fetch(`${getApiUrl()}/auth/me`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Session expired");
  }

  return data;
}
