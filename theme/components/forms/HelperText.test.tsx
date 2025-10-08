import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { HelperText } from "./HelperText";

describe("HelperText", () => {
  it("renders helper content", async () => {
    const { container } = render(
      <HelperText id="email-helper">Hilfreicher Hinweis</HelperText>
    );

    const text = screen.getByText("Hilfreicher Hinweis");
    expect(text).toHaveAttribute("id", "email-helper");
    expect(await axe(container)).toHaveNoViolations();
  });
});
