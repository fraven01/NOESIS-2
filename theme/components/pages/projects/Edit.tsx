import * as React from "react";
import { Form } from "../../forms/Form";
import { FormField } from "../../forms/FormField";
import { useFormErrors } from "../../forms/useFormErrors";
import { Input } from "../../ui/Input";
import { Textarea } from "../../ui/Textarea";
import { Button } from "../../ui/Button";

/**
 * Project edit form with validation and sticky actions.
 * @example
 * <Edit initial={{name:"Alpha"}} onSubmit={console.log} />
 */
export interface EditValues {
  name: string;
  description: string;
}

export interface EditProps {
  initial?: Partial<EditValues>;
  onSubmit?: (values: EditValues) => Promise<void> | void;
  initialErrors?: Record<string, string>;
}

export const Edit: React.FC<EditProps> = ({
  initial = {},
  onSubmit,
  initialErrors = {},
}) => {
  const { errors, setErrors, setServerErrors } = useFormErrors();
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    if (Object.keys(initialErrors).length) {
      setErrors(initialErrors);
    }
  }, [initialErrors, setErrors]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    const values: EditValues = {
      name: (formData.get("name") as string)?.trim() || "",
      description: (formData.get("description") as string)?.trim() || "",
    };
    const newErrors: Record<string, string> = {};
    if (!values.name) {
      newErrors.name = "Name is required";
    }
    if (Object.keys(newErrors).length) {
      setErrors(newErrors);
      return;
    }
    setLoading(true);
    try {
      await onSubmit?.(values);
    } catch (err: any) {
      if (err && typeof err === "object" && "errors" in err) {
        setServerErrors((err as any).errors);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form onSubmit={handleSubmit} className="min-h-screen">
      <div className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-2">
        <FormField
          name="name"
          label="Name"
          helperText="Project name must be unique"
          error={errors.name}
        >
          <Input defaultValue={initial.name} className="h-11" />
        </FormField>
        <FormField
          name="description"
          label="Description"
          helperText="Optional"
          error={errors.description}
        >
          <Textarea defaultValue={initial.description} className="min-h-11" />
        </FormField>
      </div>
      <div className="sticky bottom-0 mt-4 flex justify-end gap-2 border-t bg-bg p-4">
        <Button type="button" variant="secondary" className="h-11 px-6">
          Cancel
        </Button>
        <Button type="submit" className="h-11 px-6" disabled={loading}>
          {loading ? "Saving..." : "Save"}
        </Button>
      </div>
    </Form>
  );
};
