import * as React from "react";
import { cn } from "./cn";

/**
 * Breadcrumb navigation
 * @example
 * <Breadcrumbs items={[{href:'/', label:'Home'}]} />
 */
export interface BreadcrumbItem {
  href: string;
  label: string;
}
export interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  className?: string;
}

export const Breadcrumbs: React.FC<BreadcrumbsProps> = ({ items, className }) => (
  <nav className={cn("text-sm", className)} aria-label="Breadcrumb">
    <ol className="flex flex-wrap gap-1">
      {items.map((item, i) => (
        <li key={item.href} className="flex items-center">
          {i > 0 && <span className="mx-1">/</span>}
          <a href={item.href} className="text-accent hover:underline">
            {item.label}
          </a>
        </li>
      ))}
    </ol>
  </nav>
);
