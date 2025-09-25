import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Card } from "./Card";

describe("Card", () => {
  it("renders children and is accessible", async () => {
    const { container } = render(
      <Card>
        <p>Content</p>
      </Card>
    );
    expect(screen.getByText("Content")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});
