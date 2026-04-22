export default {
    content: [
        './index.html',
        './src/**/*.{js,jsx,ts,tsx}',
    ],
    theme: {
        extend: {
            keyframes: {
                wave: {
                    '0%, 100%': { transform: 'scaleY(0.4)' },
                    '50%': { transform: 'scaleY(1.5)' },
                },
            },
            animation: {
                wave: 'wave 1s ease-in-out infinite',
            },
        },
    },
    plugins: [],
};