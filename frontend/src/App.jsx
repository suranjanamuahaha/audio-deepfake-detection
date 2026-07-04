import "./App.css";

import { Routes, Route } from "react-router-dom";

import { NavBar } from "./components/Navbar";
import { Hero } from "./components/Hero";

import Login from "./pages/Login";
import SpamHistory from "./pages/SpamHistory";

import ProtectedRoute from "./components/ProtectedRoute";

function App() {
  return (
    <>
      <NavBar />

      <Routes>
        <Route path="/" element={<Hero />} />

        <Route path="/login" element={<Login />} />

        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <SpamHistory />
            </ProtectedRoute>
          }
        />
      </Routes>
    </>
  );
}

export default App;