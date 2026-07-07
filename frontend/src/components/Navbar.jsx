import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export const NavBar = () => {
    const navigate = useNavigate();
    const { user, logout } = useAuth();

    const navLinks = [
        { name: "Home", path: "/" },
        { name: "Process", id: "process" },
        { name: "About", id: "about" },
    ];

    const [isScrolled, setIsScrolled] = useState(false);
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleLinkClick = (e, link) => {
        if (link.id) {
            e.preventDefault();
            const element = document.getElementById(link.id);
            if (element) {
                element.scrollIntoView({ behavior: "smooth" });
            }
        }
    };

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 10);
        };

        window.addEventListener("scroll", handleScroll);
        handleScroll();

        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    const handleLogout = () => {
        logout();
        navigate("/");
    };

    return (
        <nav
            className={`fixed top-0 left-0 w-full flex items-center justify-between px-4 md:px-16 lg:px-24 xl:px-32 transition-all duration-500 z-50 ${
                isScrolled
                    ? "bg-white/80 shadow-md text-gray-700 backdrop-blur-lg py-3 md:py-4"
                    : "bg-transparent py-4 md:py-6"
            }`}
        >
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
                <img
                    src="/logo1.jpeg"
                    alt="Logo"
                    className={`h-8 md:h-10 cursor-pointer transition-all duration-300 ${
                        isScrolled ? "invert" : ""
                    }`}
                />

                <span
                    className={`text-sm md:text-lg font-semibold tracking-wide transition-all duration-300 ${
                        isScrolled ? "text-black" : "text-white/90"
                    }`}
                >
                    DATADEFENDERS
                </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-4 lg:gap-8 absolute left-1/2 transform -translate-x-1/2">
                {navLinks.map((link, i) => (
                    <a
                        key={i}
                        href={link.path || "#"}
                        onClick={(e) => handleLinkClick(e, link)}
                        className={`group flex flex-col gap-0.5 cursor-pointer ${
                            isScrolled ? "text-gray-700" : "text-white"
                        }`}
                    >
                        {link.name}

                        <div
                            className={`${
                                isScrolled ? "bg-gray-700" : "bg-white"
                            } h-0.5 w-0 group-hover:w-full transition-all duration-300`}
                        />
                    </a>
                ))}
            </div>

            {/* Desktop Auth Buttons */}
            <div className="hidden md:flex items-center gap-4">
                {!user ? (
                    <Link
                        to="/login"
                        className="bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-lg transition"
                    >
                        Login
                    </Link>
                ) : (
                    <>
                        <Link
                            to="/history"
                            className={`${
                                isScrolled ? "text-gray-700" : "text-white"
                            } hover:text-red-500 transition`}
                        >
                            History
                        </Link>

                        <button
                            onClick={handleLogout}
                            className="bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-lg transition"
                        >
                            Logout
                        </button>
                    </>
                )}
            </div>

            {/* Mobile Menu Button */}
            <div className="flex items-center gap-3 md:hidden ml-auto">
                <svg
                    onClick={() => setIsMenuOpen(!isMenuOpen)}
                    className={`h-6 w-6 cursor-pointer ${
                        isScrolled ? "invert" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                >
                    <line x1="4" y1="6" x2="20" y2="6" />
                    <line x1="4" y1="12" x2="20" y2="12" />
                    <line x1="4" y1="18" x2="20" y2="18" />
                </svg>
            </div>

            {/* Mobile Menu */}
            <div
                className={`fixed top-0 left-0 w-full h-screen bg-white text-base flex flex-col md:hidden items-center justify-center gap-6 font-medium text-gray-800 transition-all duration-500 ${
                    isMenuOpen ? "translate-x-0" : "-translate-x-full"
                }`}
            >
                <button
                    className="absolute top-4 right-4"
                    onClick={() => setIsMenuOpen(false)}
                >
                    <svg
                        className="h-6 w-6"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        viewBox="0 0 24 24"
                    >
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                </button>

                {navLinks.map((link, i) => (
                    <a
                        key={i}
                        href={link.path || "#"}
                        onClick={(e) => {
                            handleLinkClick(e, link);
                            setIsMenuOpen(false);
                        }}
                    >
                        {link.name}
                    </a>
                ))}

                {!user ? (
                    <Link
                        to="/login"
                        onClick={() => setIsMenuOpen(false)}
                        className="bg-red-600 text-white px-5 py-2 rounded-lg"
                    >
                        Login
                    </Link>
                ) : (
                    <>
                        <Link
                            to="/history"
                            onClick={() => setIsMenuOpen(false)}
                        >
                            History
                        </Link>

                        <button
                            onClick={() => {
                                handleLogout();
                                setIsMenuOpen(false);
                            }}
                            className="bg-red-600 text-white px-5 py-2 rounded-lg"
                        >
                            Logout
                        </button>
                    </>
                )}
            </div>
        </nav>
    );
};