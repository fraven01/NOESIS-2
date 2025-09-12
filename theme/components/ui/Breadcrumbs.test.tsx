import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Breadcrumbs } from "./Breadcrumbs";

describe("Breadcrumbs", () => {
  it("marks last item as current and is accessible", async () => {
    const items = [
      { href: "/", label: "Home" },
      { href: "/docs", label: "Docs" },
    ];
    const { container } = render(<Breadcrumbs items={items} />);
    expect(
      screen.getByRole("navigation", { name: /breadcrumb/i })
    ).toBeInTheDocument();
    const home = screen.getByRole("link", { name: "Home" });
    expect(home).toHaveAttribute("href", "/");
    const docs = screen.getByText("Docs");
    expect(docs).toHaveAttribute("aria-current", "page");
    expect(await axe(container)).toHaveNoViolations();
  });
});
