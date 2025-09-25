import * as React from "react";

/**
 * Manages client and server validation errors.
 * @example
 * const { errors, setErrors, setServerErrors } = useFormErrors();
 */
export const useFormErrors = () => {
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const setServerErrors = (errs: Record<string, string[] | string>) => {
    const flat = Object.fromEntries(
      Object.entries(errs).map(([k, v]) => [k, Array.isArray(v) ? v.join(" ") : v])
    );
    setErrors(flat);
  };
  return { errors, setErrors, setServerErrors };
};
