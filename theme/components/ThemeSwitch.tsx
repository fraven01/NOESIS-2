import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { cn } from "./ui/cn";
import { useTheme } from "./hooks/useTheme";

/**
 * Toggles between light and dark themes.
 * @example
 * <ThemeSwitch />
 */
export interface ThemeSwitchProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {}

export const ThemeSwitch = React.forwardRef<HTMLButtonElement, ThemeSwitchProps>(
  ({ className, ...props }, ref) => {
    const { theme, toggle } = useTheme();
    const iconClasses = "h-5 w-5";
    return (
      <button
        type="button"
        ref={ref}
        aria-label="Toggle theme"
        onClick={toggle}
        className={cn(
          "rounded p-2 text-fg transition-colors hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-accent",
          className
        )}
        {...props}
      >
        {theme === "dark" ? <Sun aria-hidden className={iconClasses} /> : <Moon aria-hidden className={iconClasses} />}
      </button>
    );
  }
);
ThemeSwitch.displayName = "ThemeSwitch";
