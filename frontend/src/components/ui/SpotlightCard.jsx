import React, { useRef, useState, useEffect } from 'react';
import { useTheme } from '@/components/ThemeProvider';
import './SpotlightCard.css';

const SpotlightCard = ({ children, className = '', spotlightColor, ...rest }) => {
  const divRef = useRef(null);
  const [activeTheme, setActiveTheme] = useState('dark');

  let contextTheme = null;
  try {
    const context = useTheme();
    contextTheme = context.theme;
  } catch (e) {
    // Ignore error if context is not available
  }

  useEffect(() => {
    if (contextTheme) {
      setActiveTheme(contextTheme);
      return;
    }

    // Fallback: detect theme dynamically using MutationObserver on html element class list
    const observer = new MutationObserver(() => {
      const isDark = document.documentElement.classList.contains('dark');
      setActiveTheme(isDark ? 'dark' : 'light');
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    // Initial check
    const isDark = document.documentElement.classList.contains('dark');
    setActiveTheme(isDark ? 'dark' : 'light');

    return () => observer.disconnect();
  }, [contextTheme]);

  const defaultSpotlightColor = activeTheme === 'light'
    ? 'rgba(186, 230, 253, 0.55)' // touch of #BAE6FD (rgb 186, 230, 253) for light mode visibility
    : 'rgba(255, 255, 255, 0.38)'; // increased dark mode spotlight brightness (by over 50%)

  const activeSpotlightColor = spotlightColor || defaultSpotlightColor;

  const handleMouseMove = e => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    divRef.current.style.setProperty('--mouse-x', `${x}px`);
    divRef.current.style.setProperty('--mouse-y', `${y}px`);
    divRef.current.style.setProperty('--spotlight-color', activeSpotlightColor);
  };

  return (
    <div ref={divRef} onMouseMove={handleMouseMove} className={`card-spotlight ${className}`} {...rest}>
      {children}
    </div>
  );
};

export default SpotlightCard;
