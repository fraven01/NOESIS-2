import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from './utils';

/**
 * Button provides a consistent trigger element with variants and sizes.
 *
 * @example
 * ```tsx
 * <Button variant="primary">Save</Button>
 * ```
 */
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded font-medium transition-colors focus:outline-none focus-visible:ring disabled:opacity-50 disabled:pointer-events-none',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-bg hover:bg-accent/90',
        secondary: 'bg-bg border border-muted text-fg hover:bg-muted',
        ghost: 'bg-transparent text-fg hover:bg-muted',
        destructive: 'bg-danger text-bg hover:bg-danger/90',
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4',
        lg: 'h-12 px-6 text-lg',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  isLoading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, isLoading, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        aria-busy={isLoading || undefined}
        disabled={disabled || isLoading}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

export default Button;
