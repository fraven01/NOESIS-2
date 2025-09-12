import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Toast } from "./Toast";
import { vi } from "vitest";

describe("Toast", () => {
  it("renders and is accessible", async () => {
    const onClose = vi.fn();
    const { container } = render(
      <Toast open onClose={onClose} autoHide={1000}>
        Saved
      </Toast>
    );
    expect(screen.getByRole("status")).toHaveTextContent("Saved");
    expect(await axe(container)).toHaveNoViolations();
  });
});
