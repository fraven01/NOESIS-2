import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { ErrorText } from "./ErrorText";

describe("ErrorText", () => {
  it("exposes alert semantics", async () => {
    const { container } = render(
      <ErrorText id="email-error">Bitte gib eine gültige E-Mail an.</ErrorText>
    );

    const text = screen.getByRole("alert");
    expect(text).toHaveAttribute("id", "email-error");
    expect(await axe(container)).toHaveNoViolations();
  });
});
