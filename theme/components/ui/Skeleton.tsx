import * as React from "react";
import { cn } from "./cn";

/**
 * Skeleton placeholder
 * @example
 * <Skeleton className="h-4 w-20" />
 */
export type SkeletonProps = React.HTMLAttributes<HTMLDivElement>;

export const Skeleton = React.forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("animate-pulse rounded-xl bg-muted", className)}
      {...props}
    />
  )
);
Skeleton.displayName = "Skeleton";
